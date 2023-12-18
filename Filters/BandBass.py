
import cv2
import numpy as np
from matplotlib import pyplot as plt

def band_pass_filter(image_path, low_cutoff = 10 , high_cutoff = 30):

	f = cv2.imread(image_path, 0)
	F = np.fft.fft2(f)
	Fshift = np.fft.fftshift(F)

	M,N = f.shape
	H = np.ones((M,N) , dtype = np.float32)

	for u in range(M):
		for v in range(N):
			D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
			if D < low_cutoff or D > high_cutoff:
                           H[u,v] = 0




	# #show the 	filter
	# plt.figure(figsize = (10,10))
	# plt.imshow(H, cmap = 'gray')
	# plt.axis('off')
	# plt.show()

	### make Low Pass on Image
	Gshift = Fshift * H
	## make inverse to return spatial domain
	G = np.fft.ifftshift(Gshift)
	g = np.abs(np.fft.ifft2(G))

	return g

