import numpy as np
import cv2
import random

def imcv2_recolor(im, a = .1):
	selection = random.randint(0,9)
	if selection < 7:
		t = [np.random.uniform()]
		t += [np.random.uniform()]
		t += [np.random.uniform()]
		t = np.array(t) * 2. - 1.

		# random amplify each channel
		im = im * (1 + t * a)
		mx = 255. * (1 + a)
		up = np.random.uniform() * 2 - 1
		# im = np.power(im/mx, 1. + up * .5)
		im = cv2.pow(im/mx, 1. + up * .5)
		return np.array(im * 255., np.uint8)
	else:
		return im

def imcv2_affine_trans(im):
	# Scale and translate
	h, w, c = im.shape
	scale = np.random.uniform() / 10. + 1.
	max_offx = (scale-1.) * w
	max_offy = (scale-1.) * h
	offx = int(np.random.uniform() * max_offx)
	offy = int(np.random.uniform() * max_offy)
	
	im = cv2.resize(im, (0,0), fx = scale, fy = scale)
	im = im[offy : (offy + h), offx : (offx + w)]
	flip = np.random.binomial(1, .5)
	vflip = np.random.binomial(1, .5)
	if vflip: im = cv2.flip(im, 0)
	if flip: im = cv2.flip(im, 1)
	return im, [w, h, c], [scale, [offx, offy], flip, vflip]

def imcv2_noise(im):
	selection = random.randint(0,3) #create a random number between 0-2, 0-gauss 1-s&p 2-speckle
	h, w, c = im.shape
	if selection == 0:
		mean = 0
		var = 0.1
		sigma = var ** 0.5
		gauss = np.random.normal(mean, sigma, (h, w, c))
		gauss = gauss.reshape(h, w, c)
		return im + gauss
	elif selection == 1:
		s_vs_p = 0.5
		amount = random.uniform(0, .05)
		size = im.size
		#Salt first
		num_salt = np.ceil(amount * size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt)) for i in (h, w, c)]
		im[coords] = 1
		#Then pepper
		num_pepper = np.ceil(amount * size * (1 - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in (h, w, c)]
		im[coords] = 0
		return im
	elif selection == 2:
		gauss = np.random.randn(h, w, c)
		gauss = gauss.reshape(h, w, c)
		noisy = im + im * gauss
		return noisy
	else:
		return im
