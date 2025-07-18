import torch
import js_cpp


class JSOptfunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, unbiased,biased, var, param, win_size):
		assert not hasattr(ctx, 'win_size') or ctx.win_size is None
		ctx._win_size = win_size
		out_img, rho = js_cpp.js_opt_forward(unbiased, biased, var, param, win_size)
		return out_img, rho

class JS_opt(torch.nn.Module):
	def __init__(self, win_size):
		super(JS_opt, self).__init__()
		self.win_size  = win_size

	def forward(self, unbiasd, biased, var, param):
		output, rho = JSOptfunction.apply(unbiasd, biased, var, param, self.win_size)
		return output, rho

