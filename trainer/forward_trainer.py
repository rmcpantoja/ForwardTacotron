import time
from typing import Dict, Any, Union

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.hifigan import MultiPeriodDiscriminator, feature_loss, discriminator_loss, generator_loss
from models.hifiutils import plot_spectrogram_to_numpy
from models.fast_pitch import FastPitch
from models.forward_tacotron import ForwardTacotron
from trainer.common import Averager, TTSSession, MaskedL1, to_device, np_now
from utils.checkpoints import  save_checkpoint
from utils.dataset import get_forward_dataloaders
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel, plot_pitch
from utils.dsp import DSP
from utils.files import parse_schedule
from utils.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from utils.paths import Paths


class ForwardTrainer:

    def __init__(self,
                 paths: Paths,
                 dsp: DSP,
                 config: Dict[str, Any]) -> None:
        self.paths = paths
        self.dsp = dsp
        self.config = config
        model_type = config.get('tts_model', 'forward_tacotron')
        self.train_cfg = config[model_type]['training']
        self.writer = SummaryWriter(log_dir=paths.forward_log, comment='v1')
        self.l1_loss = MaskedL1()

    def train(self, model: Union[ForwardTacotron, FastPitch], disc: MultiPeriodDiscriminator, optimizers: list[Optimizer], schedulers: list, scaler: GradScaler) -> None:
        forward_schedule = self.train_cfg['schedule']
        forward_schedule = parse_schedule(forward_schedule)
        for i, session_params in enumerate(forward_schedule, 1):
            lr, max_step, bs = session_params
            filter_params = self.train_cfg['filter']
            if model.get_step() < max_step:
                train_set, val_set = get_forward_dataloaders(
                    paths=self.paths, batch_size=bs, **filter_params)
                session = TTSSession(
                    index=i, r=1, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, disc, optimizers, schedulers, scaler, session)

    def train_session(self,  model: Union[ForwardTacotron, FastPitch], disc: MultiPeriodDiscriminator,
                      optimizers: list[Optimizer], schedulers: list, scaler: GradScaler, session: TTSSession) -> None:
        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        simple_table([(f'Steps', str(training_steps // 1000) + 'k Steps'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr)])
        optim_g, optim_d = optimizers
        scheduler_g, scheduler_d = schedulers
        for g in optim_g.param_groups:
            g['lr'] = session.lr

        m_loss_avg = Averager()
        dur_loss_avg = Averager()
        duration_avg = Averager()
        pitch_loss_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters
        for e in range(1, epochs + 1):
            for i, batch in enumerate(session.train_set, 1):
                batch = to_device(batch, device=device)
                start = time.time()
                model.train()
                disc.train()
                pitch_zoneout_mask = torch.rand(batch['x'].size()) > self.train_cfg['pitch_zoneout']
                energy_zoneout_mask = torch.rand(batch['x'].size()) > self.train_cfg['energy_zoneout']

                pitch_target = batch['pitch'].detach().clone()
                energy_target = batch['energy'].detach().clone()
                batch['pitch'] = batch['pitch'] * pitch_zoneout_mask.to(device).float()
                batch['energy'] = batch['energy'] * energy_zoneout_mask.to(device).float()

                pred = model(batch)
                pred['audio'] =                 pred['audio'].float()
                hat_mel = mel_spectrogram_torch(
                    pred['audio'].squeeze(1), 
                    self.config["dsp"]["win_length"], 
                    self.config["dsp"]["num_mels"], 
                    self.config["dsp"]["sample_rate"], 
                    self.config["dsp"]["hop_length"], 
                    self.config["dsp"]["win_length"], 
                    self.config["dsp"]["fmin"], 
                    self.config["dsp"]["fmax"]
                )
                y_d_hat_r, y_d_hat_g, _, _ = disc(batch['wav'], pred['audio'].detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    loss_disc_all = loss_disc
                optim_d.zero_grad()
                scaler.scale(loss_disc_all).backward()
                scaler.unscale_(optim_d)
                grad_norm_d = commons.clip_grad_value_(disc.parameters(), None)
                scaler.step(optim_d)
                m1_loss = self.l1_loss(pred['mel'], batch['mel'], batch['mel_len'])
                m2_loss = self.l1_loss(pred['mel_post'], batch['mel'], batch['mel_len'])

                dur_loss = self.l1_loss(pred['dur'].unsqueeze(1), batch['dur'].unsqueeze(1), batch['x_len'])
                pitch_loss = self.l1_loss(pred['pitch'], pitch_target.unsqueeze(1), batch['x_len'])
                energy_loss = self.l1_loss(pred['energy'], energy_target.unsqueeze(1), batch['x_len'])

                loss = m1_loss + m2_loss \
                       + self.train_cfg['dur_loss_factor'] * dur_loss \
                       + self.train_cfg['pitch_loss_factor'] * pitch_loss \
                       + self.train_cfg['energy_loss_factor'] * energy_loss
                with autocast(enabled=hps.train.fp16_run):
                    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = disc(batch['wav'], y_hat)
                    with autocast(enabled=False):
                        loss_mel = F.l1_loss(batch['wav'], y_hat_mel) * 45
                        loss_fm = feature_loss(fmap_r, fmap_g)
                        loss_gen, losses_gen = generator_loss(y_d_hat_g)
                        loss_gen_all = loss_gen + loss_fm + loss_mel
                optim_g.zero_grad()
                scaler.scale(loss_gen_all).backward()
                loss.backward()
                scaler.unscale_(optim_g)
                grad_norm_g = commons.clip_grad_value_(model.parameters(), None)
                scaler.step(optim_g)
                scaler.update()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               self.train_cfg['clip_grad_norm'])
                optim_g.step()

                m_loss_avg.add(m1_loss.item() + m2_loss.item())
                dur_loss_avg.add(dur_loss.item())
                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                pitch_loss_avg.add(pitch_loss.item())

                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Mel Loss: {m_loss_avg.get():#.4} ' \
                      f'| Dur Loss: {dur_loss_avg.get():#.4} | Pitch Loss: {pitch_loss_avg.get():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                if step % self.train_cfg['checkpoint_every'] == 0:
                    lr = optim_g.param_groups[0]['lr']
                    scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
                    scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel})
                    scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                    scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                    scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                    save_checkpoint(model=model, optim=optim_g, config=self.config,
                                    path=self.paths.forward_checkpoints / f'g_{k}k.pt')
                    save_checkpoint(model=disc, optim=optim_d, config=self.config,
                                    path=self.paths.forward_checkpoints / f'd{k}k.pt')

                if step % self.train_cfg['plot_every'] == 0:
                    self.generate_plots(model, session)

                self.writer.add_scalar('Mel_Loss/train', m1_loss + m2_loss, model.get_step())
                self.writer.add_scalar('Pitch_Loss/train', pitch_loss, model.get_step())
                self.writer.add_scalar('Energy_Loss/train', energy_loss, model.get_step())
                self.writer.add_scalar('Duration_Loss/train', dur_loss, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            val_out = self.evaluate(model, session.val_set)
            self.writer.add_scalar('Mel_Loss/val', val_out['mel_loss'], model.get_step())
            self.writer.add_scalar('Duration_Loss/val', val_out['dur_loss'], model.get_step())
            self.writer.add_scalar('Pitch_Loss/val', val_out['pitch_loss'], model.get_step())
            self.writer.add_scalar('Energy_Loss/val', val_out['energy_loss'], model.get_step())
            save_checkpoint(model=model, optim=optim_g, config=self.config,
                            path=self.paths.forward_checkpoints / 'latest_g.pt')
            save_checkpoint(model=disc, optim=optim_d, config=self.config,
                            path=self.paths.forward_checkpoints / 'latest_d.pt')

            m_loss_avg.reset()
            duration_avg.reset()
            pitch_loss_avg.reset()
            print(' ')

    def evaluate(self, model: Union[ForwardTacotron, FastPitch], val_set: DataLoader) -> Dict[str, float]:
        model.eval()
        m_val_loss = 0
        dur_val_loss = 0
        pitch_val_loss = 0
        energy_val_loss = 0
        device = next(model.parameters()).device
        for i, batch in enumerate(val_set, 1):
            batch = to_device(batch, device=device)
            with torch.no_grad():
                pred = model(batch)
                m1_loss = self.l1_loss(pred['mel'], batch['mel'], batch['mel_len'])
                m2_loss = self.l1_loss(pred['mel_post'], batch['mel'], batch['mel_len'])
                dur_loss = self.l1_loss(pred['dur'].unsqueeze(1), batch['dur'].unsqueeze(1), batch['x_len'])
                pitch_loss = self.l1_loss(pred['pitch'], batch['pitch'].unsqueeze(1), batch['x_len'])
                energy_loss = self.l1_loss(pred['energy'], batch['energy'].unsqueeze(1), batch['x_len'])
                y_hat_mel = mel_spectrogram_torch(
                    pred['audio'].squeeze(1), 
                    self.config["dsp"]["win_length"], 
                    self.config["dsp"]["num_mels"], 
                    self.config["dsp"]["sample_rate"], 
                    self.config["dsp"]["hop_length"], 
                    self.config["dsp"]["win_length"], 
                    self.config["dsp"]["fmin"], 
                    self.config["dsp"]["fmax"]
                )
                e2e_loss = self.l1_loss(y_hat_mel, batch['wav'].unsqueeze(1), batch['wav_length'])
                pitch_val_loss += pitch_loss
                energy_val_loss += energy_loss
                m_val_loss += m1_loss.item() + m2_loss.item()
                dur_val_loss += dur_loss.item()
                e2e_val_loss += e2e_loss.item()
        return {
            'mel_loss': m_val_loss / len(val_set),
            'mel_e2e_loss': e2e_val_loss / len(val_set),
            'dur_loss': dur_val_loss / len(val_set),
            'pitch_loss': pitch_val_loss / len(val_set),
            'energy_loss': energy_val_loss / len(val_set)
        }

    @ignore_exception
    def generate_plots(self, model: Union[ForwardTacotron, FastPitch], session: TTSSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        batch = session.val_sample
        batch = to_device(batch, device=device)

        pred = model(batch)
        m1_hat = np_now(pred['mel'])[0, :, :]
        m2_hat = np_now(pred['mel_post'])[0, :, :]
        audio_hat = np_now(pred['audio'])[0, :, :]
        m_target = np_now(batch['mel'])[0, :, :]

        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)
        audio_hat_fig = plot_spectrogram_to_numpy(audio_hat)
        m_target_fig = plot_mel(m_target)
        pitch_fig = plot_pitch(np_now(batch['pitch'][0]))
        pitch_gta_fig = plot_pitch(np_now(pred['pitch'].squeeze()[0]))
        energy_fig = plot_pitch(np_now(batch['energy'][0]))
        energy_gta_fig = plot_pitch(np_now(pred['energy'].squeeze()[0]))

        self.writer.add_figure('Pitch/target', pitch_fig, model.step)
        self.writer.add_figure('Pitch/ground_truth_aligned', pitch_gta_fig, model.step)
        self.writer.add_figure('Energy/target', energy_fig, model.step)
        self.writer.add_figure('Energy/ground_truth_aligned', energy_gta_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/target', m_target_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/linear', m1_hat_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/postnet', m2_hat_fig, model.step)
        self.writer.add_figure('Ground_Truth_Aligned/audio', audio_hat_fig, model.step)
        target_wav = self.dsp.griffinlim(m_target)

        self.writer.add_audio(
            tag='Ground_Truth_Aligned/target_wav', snd_tensor=target_wav,
            global_step=model.step, sample_rate=self.dsp.sample_rate)
        self.writer.add_audio(
            tag='Ground_Truth_Aligned/postnet_wav', snd_tensor=audio_hat,
            global_step=model.step, sample_rate=self.dsp.sample_rate)

        gen = model.generate(batch['x'][0:1, :batch['x_len'][0]])
        m1_hat = np_now(gen['mel'].squeeze())
        m2_hat = np_now(gen['mel_post'].squeeze())
        audio_hat = np_now(gen['audio'].squeeze())
        m1_hat_fig = plot_mel(m1_hat)
        m2_hat_fig = plot_mel(m2_hat)
        audio_hat_fig = plot_spectrogram_to_numpy(audio_hat)
        pitch_gen_fig = plot_pitch(np_now(gen['pitch'].squeeze()))
        energy_gen_fig = plot_pitch(np_now(gen['energy'].squeeze()))

        self.writer.add_figure('Pitch/generated', pitch_gen_fig, model.step)
        self.writer.add_figure('Energy/generated', energy_gen_fig, model.step)
        self.writer.add_figure('Generated/target', m_target_fig, model.step)
        self.writer.add_figure('Generated/linear', m1_hat_fig, model.step)
        self.writer.add_figure('Generated/postnet', m2_hat_fig, model.step)
        self.writer.add_figure('Generated/audio', audio_hat_fig, model.step)

        self.writer.add_audio(
            tag='Generated/target_wav', snd_tensor=target_wav,
            global_step=model.step, sample_rate=self.dsp.sample_rate)
        self.writer.add_audio(
            tag='Generated/postnet_wav', snd_tensor=audio_hat,
            global_step=model.step, sample_rate=self.dsp.sample_rate)
