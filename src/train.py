import sys
import time

import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src import ctc, models
from src.eval_metrics import eval_iemocap, eval_mosei_senti, eval_mosi
from src.utils import load_model, save_model


def get_CTC_module(hyp_params):
    a2l_module = getattr(ctc, "CTCModule")(in_dim=hyp_params.orig_d_a, out_seq_len=hyp_params.l_len)
    v2l_module = getattr(ctc, "CTCModule")(in_dim=hyp_params.orig_d_v, out_seq_len=hyp_params.l_len)
    return a2l_module, v2l_module


def maybe_data_parallel(module, enabled):
    if module is None:
        return None
    return nn.DataParallel(module) if enabled else module


def move_batch_to_device(batch_X, batch_Y, hyp_params):
    _, text, audio, vision = batch_X
    eval_attr = batch_Y.squeeze(dim=-1)
    if hyp_params.dataset == "iemocap":
        eval_attr = eval_attr.long()

    if hyp_params.use_cuda:
        text = text.to(hyp_params.device, non_blocking=True)
        audio = audio.to(hyp_params.device, non_blocking=True)
        vision = vision.to(hyp_params.device, non_blocking=True)
        eval_attr = eval_attr.to(hyp_params.device, non_blocking=True)
    else:
        text = text.to(hyp_params.device)
        audio = audio.to(hyp_params.device)
        vision = vision.to(hyp_params.device)
        eval_attr = eval_attr.to(hyp_params.device)
    return text, audio, vision, eval_attr


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model_cls = getattr(models, hyp_params.model + "Model")
    model = model_cls(hyp_params).to(hyp_params.device)
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    use_data_parallel = hyp_params.use_cuda and torch.cuda.device_count() > 1

    if hyp_params.aligned or hyp_params.model == "MULT":
        ctc_criterion = None
        ctc_a2l_module = None
        ctc_v2l_module = None
        ctc_a2l_optimizer = None
        ctc_v2l_optimizer = None
    else:
        from warpctc_pytorch import CTCLoss

        ctc_criterion = CTCLoss()
        ctc_a2l_module, ctc_v2l_module = get_CTC_module(hyp_params)
        ctc_a2l_module = ctc_a2l_module.to(hyp_params.device)
        ctc_v2l_module = ctc_v2l_module.to(hyp_params.device)
        ctc_a2l_optimizer = getattr(optim, hyp_params.optim)(ctc_a2l_module.parameters(), lr=hyp_params.lr)
        ctc_v2l_optimizer = getattr(optim, hyp_params.optim)(ctc_v2l_module.parameters(), lr=hyp_params.lr)

    settings = {
        "model": model,
        "parallel_model": maybe_data_parallel(model, use_data_parallel),
        "optimizer": optimizer,
        "criterion": criterion,
        "ctc_a2l_module": ctc_a2l_module,
        "ctc_v2l_module": ctc_v2l_module,
        "parallel_ctc_a2l_module": maybe_data_parallel(ctc_a2l_module, use_data_parallel),
        "parallel_ctc_v2l_module": maybe_data_parallel(ctc_v2l_module, use_data_parallel),
        "ctc_a2l_optimizer": ctc_a2l_optimizer,
        "ctc_v2l_optimizer": ctc_v2l_optimizer,
        "ctc_criterion": ctc_criterion,
        "scheduler": ReduceLROnPlateau(
            optimizer, mode="min", patience=hyp_params.when, factor=0.1, verbose=True
        ),
        "model_cls": model_cls,
    }
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings["model"]
    parallel_model = settings["parallel_model"]
    optimizer = settings["optimizer"]
    criterion = settings["criterion"]
    ctc_a2l_module = settings["ctc_a2l_module"]
    ctc_v2l_module = settings["ctc_v2l_module"]
    parallel_ctc_a2l_module = settings["parallel_ctc_a2l_module"]
    parallel_ctc_v2l_module = settings["parallel_ctc_v2l_module"]
    ctc_a2l_optimizer = settings["ctc_a2l_optimizer"]
    ctc_v2l_optimizer = settings["ctc_v2l_optimizer"]
    ctc_criterion = settings["ctc_criterion"]
    scheduler = settings["scheduler"]
    model_cls = settings["model_cls"]
    grad_accum_steps = hyp_params.grad_accum_steps
    overall_start = time.time()
    use_amp = hyp_params.use_amp and ctc_criterion is None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if hyp_params.use_cuda:
        torch.cuda.reset_peak_memory_stats(hyp_params.device)

    def zero_all_grads():
        optimizer.zero_grad(set_to_none=True)
        if ctc_criterion is not None:
            ctc_a2l_optimizer.zero_grad(set_to_none=True)
            ctc_v2l_optimizer.zero_grad(set_to_none=True)

    def step_all_optimizers():
        if ctc_criterion is not None:
            torch.nn.utils.clip_grad_norm_(ctc_a2l_module.parameters(), hyp_params.clip)
            torch.nn.utils.clip_grad_norm_(ctc_v2l_module.parameters(), hyp_params.clip)
            ctc_a2l_optimizer.step()
            ctc_v2l_optimizer.step()
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        zero_all_grads()

    def compute_ctc_loss(audio, vision, batch_size):
        if ctc_criterion is None:
            return audio, vision, None

        audio, a2l_position = parallel_ctc_a2l_module(audio)
        vision, v2l_position = parallel_ctc_v2l_module(vision)

        l_len, a_len, v_len = hyp_params.l_len, hyp_params.a_len, hyp_params.v_len
        l_position = torch.tensor([i + 1 for i in range(l_len)] * batch_size).int().cpu()
        l_length = torch.tensor([l_len] * batch_size).int().cpu()
        a_length = torch.tensor([a_len] * batch_size).int().cpu()
        v_length = torch.tensor([v_len] * batch_size).int().cpu()

        ctc_a2l_loss = ctc_criterion(a2l_position.transpose(0, 1).cpu(), l_position, a_length, l_length)
        ctc_v2l_loss = ctc_criterion(v2l_position.transpose(0, 1).cpu(), l_position, v_length, l_length)
        return audio, vision, (ctc_a2l_loss + ctc_v2l_loss).to(hyp_params.device)

    def train_epoch(epoch):
        model.train()
        zero_all_grads()
        num_batches = len(train_loader)
        epoch_loss = 0.0
        proc_loss = 0.0
        proc_size = 0
        start_time = time.time()

        for i_batch, (batch_X, batch_Y, _) in enumerate(train_loader):
            text, audio, vision, eval_attr = move_batch_to_device(batch_X, batch_Y, hyp_params)
            batch_size = text.size(0)
            audio, vision, ctc_loss = compute_ctc_loss(audio, vision, batch_size)
            ctc_loss_for_log = (
                ctc_loss.detach() if ctc_loss is not None else torch.tensor(0.0, device=hyp_params.device)
            )

            if hyp_params.batch_chunk > 1:
                raw_loss = torch.tensor(0.0, device=hyp_params.device)
                text_chunks = text.chunk(hyp_params.batch_chunk, dim=0)
                audio_chunks = audio.chunk(hyp_params.batch_chunk, dim=0)
                vision_chunks = vision.chunk(hyp_params.batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(hyp_params.batch_chunk, dim=0)

                for chunk_idx in range(hyp_params.batch_chunk):
                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        preds_i, _ = parallel_model(
                            text_chunks[chunk_idx], audio_chunks[chunk_idx], vision_chunks[chunk_idx]
                        )
                        eval_attr_i = eval_attr_chunks[chunk_idx]
                        if hyp_params.dataset == "iemocap":
                            preds_i = preds_i.view(-1, 2)
                            eval_attr_i = eval_attr_i.view(-1)
                        raw_loss_i = criterion(preds_i, eval_attr_i) / hyp_params.batch_chunk
                    raw_loss += raw_loss_i.detach()
                    scaler.scale(raw_loss_i / grad_accum_steps).backward()

                if ctc_loss is not None:
                    scaler.scale(ctc_loss / grad_accum_steps).backward()
                combined_loss = raw_loss + ctc_loss_for_log
            else:
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    preds, _ = parallel_model(text, audio, vision)
                    if hyp_params.dataset == "iemocap":
                        preds = preds.view(-1, 2)
                        eval_attr = eval_attr.view(-1)
                    raw_loss = criterion(preds, eval_attr)
                combined_loss = raw_loss + (ctc_loss if ctc_loss is not None else 0.0)
                scaler.scale(combined_loss / grad_accum_steps).backward()

            should_step = ((i_batch + 1) % grad_accum_steps == 0) or ((i_batch + 1) == num_batches)
            if should_step:
                step_all_optimizers()

            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += float(combined_loss.detach().item()) * batch_size

            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print(
                    "Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}".format(
                        epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss
                    )
                )
                proc_loss = 0.0
                proc_size = 0
                start_time = time.time()

        return epoch_loss / hyp_params.n_train

    def evaluate(test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        results = []
        truths = []

        with torch.no_grad():
            for batch_X, batch_Y, _ in loader:
                text, audio, vision, eval_attr = move_batch_to_device(batch_X, batch_Y, hyp_params)
                batch_size = text.size(0)

                if parallel_ctc_a2l_module is not None and parallel_ctc_v2l_module is not None:
                    audio, _ = parallel_ctc_a2l_module(audio)
                    vision, _ = parallel_ctc_v2l_module(vision)

                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    preds, _ = parallel_model(text, audio, vision)
                    if hyp_params.dataset == "iemocap":
                        preds = preds.view(-1, 2)
                        eval_attr = eval_attr.view(-1)
                    loss = criterion(preds, eval_attr)
                total_loss += loss.item() * batch_size
                results.append(preds.detach())
                truths.append(eval_attr.detach())

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        return avg_loss, torch.cat(results), torch.cat(truths)

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train_epoch(epoch)
        val_loss, _, _ = evaluate(test=False)
        test_loss, _, _ = evaluate(test=True)
        duration = time.time() - start
        scheduler.step(val_loss)

        print("-" * 50)
        print(
            "Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}".format(
                epoch, duration, val_loss, test_loss
            )
        )
        print("-" * 50)

        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    model = load_model(hyp_params, model_cls, name=hyp_params.name)
    parallel_model = maybe_data_parallel(model, hyp_params.use_cuda and torch.cuda.device_count() > 1)
    _, results, truths = evaluate(test=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == "mosi":
        eval_mosi(results, truths, True)
    elif hyp_params.dataset == "iemocap":
        eval_iemocap(results, truths)

    total_time = time.time() - overall_start
    print("Total training time (s):", total_time)
    if hyp_params.use_cuda:
        peak_memory_mb = torch.cuda.max_memory_allocated(hyp_params.device) / (1024 ** 2)
        print("Peak CUDA memory allocated (MB):", peak_memory_mb)

    sys.stdout.flush()
    if not hyp_params.no_prompt:
        input("[Press Any Key to start another run]")
