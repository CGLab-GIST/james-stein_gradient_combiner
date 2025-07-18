import torch
import filter_cpp

class AtrousFilterFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, param, numIter):
		assert not hasattr(ctx, 'numIter') or ctx.numIter is None
		ctx.numIter = numIter
		output = filter_cpp.atrous_filter(input, param, numIter)
		# ctx.save_for_backward(input, albedo, normal, depth, outWgtsum)
		return output
	
class AtrousFilter(torch.nn.Module):
	def __init__(self, numIter):
		super(AtrousFilter, self).__init__()
		self.numIter  = numIter
	def forward(self, input, param):
		output = AtrousFilterFunction.apply(input, param, self.numIter)
		return output
	

class Atrous1DFilterFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, param, numIter):
		assert not hasattr(ctx, 'numIter') or ctx.numIter is None
		ctx.numIter = numIter
		output = filter_cpp.atrous_1D_filter(input, param, numIter)
		# ctx.save_for_backward(input, albedo, normal, depth, outWgtsum)
		return output
	
class Atrous1DFilter(torch.nn.Module):
	def __init__(self, numIter):
		super(Atrous1DFilter, self).__init__()
		self.numIter  = numIter
	def forward(self, input, param):
		output = Atrous1DFilterFunction.apply(input, param, self.numIter)
		return output
	
class GaussianFilterFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, winSize):
		ctx.winSize = winSize
		output = filter_cpp.gaussian_filter(input, winSize)
		# ctx.save_for_backward(input, albedo, normal, depth, outWgtsum)
		return output
	
class GaussianFilter(torch.nn.Module):
	def __init__(self, winSize):
		super(GaussianFilter, self).__init__()
		self.winSize  = winSize
	def forward(self, input):
		output = GaussianFilterFunction.apply(input, self.winSize)
		return output
	
class Gaussian1DFilterFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, winSize):
		ctx.winSize = winSize
		output = filter_cpp.gaussian_1D_filter(input, winSize)
		# ctx.save_for_backward(input, albedo, normal, depth, outWgtsum)
		return output
	
class Gaussian1DFilter(torch.nn.Module):
	def __init__(self, winSize):
		super(Gaussian1DFilter, self).__init__()
		self.winSize  = winSize
	def forward(self, input):
		output = Gaussian1DFilterFunction.apply(input, self.winSize)
		return output