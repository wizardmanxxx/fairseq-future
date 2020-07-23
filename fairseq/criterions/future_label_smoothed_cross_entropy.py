# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch.nn.functional as F


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('future_label_smoothed_cross_entropy')
class FutureLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.future_lambda = args.future_lambda
        self.start_mse_epoch = args.start_mse_epoch
        self.forward_lambda = args.forward_lambda
        self.warmup_steps = args.future_warmup_steps
        self.use_update_mse = args.use_update_mse
        self.start_mse_updates = args.start_mse_updates

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--future-lambda', default=0.5, type=float, metavar='D',
                            help='lambda for MSE(mean square error)')
        parser.add_argument('--start-mse-epoch', default=120, type=int, metavar='D',
                            help='use x epoch to linear ')
        parser.add_argument('--use-update-mse', action='store_true', help='use update for mse')
        parser.add_argument('--start-mse-updates', default=10, type=int, metavar='D',
                            help='start mse updates number')
        parser.add_argument('--forward-lambda', default=1, type=float, metavar='D',
                            help='lambda forward loss,1-lamb backward loss')
        parser.add_argument('--future-warmup-steps', default=2000, type=int, metavar='D',
                            help='warmup for future lambda ')
        # fmt: on

    def get_future_lambda(self, epoch, num_updates):
        if self.use_update_mse and num_updates >= self.start_mse_updates:
            return self.future_lambda
        elif self.use_update_mse and num_updates < self.start_mse_updates:
            return 0

        if epoch > self.start_mse_epoch and self.warmup_steps == 0:
            return self.future_lambda
        if self.warmup_steps > 0 and num_updates != 0:
            coef = (self.warmup_steps ** 0.5)
            a = num_updates ** -0.5
            b = num_updates * (self.warmup_steps ** -1.5)
            return coef * a if a < b else coef * b
        return 0

    def forward(self, model, sample, reduce=True, num_updates=0):
        # 反向的decoder output，和正向3+3的decoder output
        net_output, net_output2 = model(**sample['net_input'])
        device = net_output[0].device
        back_loss, back_nll_loss = torch.tensor(0).to(device), torch.tensor(0).to(device)
        if self.forward_lambda < 1:
            back_loss, back_nll_loss = self.compute_back_loss(model, net_output, sample,
                                                              reduce=reduce)
        epoch = model.get_epoch()
        forward_loss, forward_nll_loss = self.compute_forward_loss(model, net_output2, sample, reduce)
        future_loss = torch.tensor(0).to(device)
        future_lambda = self.get_future_lambda(epoch, num_updates)
        if future_lambda != 0:
            future_loss = self.compute_future_loss(net_output, net_output2, sample)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']

        loss = (1 - self.forward_lambda) * back_loss + self.forward_lambda * forward_loss + future_lambda * future_loss
        # loss = forward_loss
        logging_output = {
            'loss': loss.data,
            'back_loss': back_loss.data,
            'back_nll_loss': back_nll_loss.data,
            'forward_loss': forward_loss.data,
            'forward_nll_loss': forward_nll_loss.data,
            'future_loss': future_loss.data,
            'future_lambda': future_lambda,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_future_loss(self, back_out, forward_out, sample):
        target = sample['target'].view(-1, 1)
        pad_mask = target.eq(self.padding_idx)
        # 反向decoder的最后一层作为future context,反向的decoder只提供target，不参与grad
        future_context = back_out[1]['inner_states'][-1].clone().detach().permute(1, 0, 2).contiguous()
        future_context = future_context.view(-1, future_context.size(-1)).to(dtype=torch.float32)
        hidden_dim = future_context.size(-1)
        # 取前三层作为预测future context的结构,第0是emb
        predict_context = forward_out[1]['inner_states'][3].permute(1, 0, 2).contiguous()
        predict_context = predict_context.view(-1, predict_context.size(-1)).to(dtype=torch.float32)
        if pad_mask.any():
            future_context.masked_fill_(pad_mask, 0.)
            predict_context.masked_fill_(pad_mask, 0.)
        # loss = F.mse_loss(predict_context, future_context) 该loss非常大，是否要先除以hidden-dim
        loss = (predict_context - future_context) ** 2
        return loss.sum() / hidden_dim

    def compute_back_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = sample['target_back'].view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def compute_forward_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        back_loss_sum = utils.item(sum(log.get('back_loss', 0) for log in logging_outputs))
        back_nll_loss_sum = utils.item(sum(log.get('back_nll_loss', 0) for log in logging_outputs))
        forward_loss_sum = utils.item(sum(log.get('forward_loss', 0) for log in logging_outputs))
        forward_nll_loss_sum = utils.item(sum(log.get('forward_nll_loss', 0) for log in logging_outputs))
        future_loss_sum = utils.item(sum(log.get('future_loss', 0) for log in logging_outputs))
        future_lambda = utils.item(sum(log.get('future_lambda', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('back_loss', back_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('back_nll_loss', back_nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('forward_loss', forward_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('forward_nll_loss', forward_nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('future_loss', future_loss_sum / ntokens / math.log(2), ntokens, round=3)
        # metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        metrics.log_scalar('future_lambda', future_lambda)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
