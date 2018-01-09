import cv2
import numpy as np

def calc_hog(im,numorient=9):

    sz = im.shape
    gr = np.zeros((sz[0], sz[1], 1), dtype = "uint8")
    gx = np.zeros((sz[0],sz[1], 1), dtype = "uint32")
    gy = np.zeros((sz[0],sz[1], 1), dtype = "uint32")

    #convert to grayscale
    cv2.cvtColor(im, cv2.COLOR_BGR2GRAY, gr)
    
    #calc gradient using sobel
    gx = cv2.Sobel(gr, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
    gy = cv2.Sobel(gr, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=1)

    #calc initial result
    hog = np.zeros((sz[0], sz[1], numorient))
    mid = numorient / 2
    for y in xrange(0, sz[0] - 1):
        for x in xrange(0, sz[1] - 1):
            angle = int(round(mid * np.arctan2(gy[y, x], gx[y, x]) / np.pi)) + mid
            magnitude = np.sqrt(gx[y, x] * gx[y, x] + gy[y, x] * gy[y, x])
            hog[y, x, angle] += magnitude

    # build integral image
    for x in xrange(1, sz[1] - 1):
        for ang in xrange(numorient):
            hog[y, x, ang] += hog[y, x - 1, ang]
    for y in xrange(1, sz[0] - 1):
        for ang in xrange(numorient):
            hog[y, x, ang] += hog[y - 1, x, ang]
    for y in xrange(1, sz[0] - 1):
        for x in xrange(1, sz[1] - 1):
            for ang in xrange(numorient):
                # tambah kiri dan atas, kurangi dengan kiri-atas
                hog[y, x, ang] += hog[y - 1, x, ang] + hog[y, x - 1, ang] - hog[y - 1, x - 1, ang]
    return hog


def calc_hog_block(hogim, r):
    numorient = hogim.shape[2]
    result = np.zeros(numorient)

    for ang in xrange(numorient):

        result[ang] = hogim[r[1], r[0], ang] + hogim[r[3], r[2], ang] - hogim[r[1], r[2], ang] - hogim[r[3], r[0], ang]

    return result

def draw_hog(target, ihog, cellsize=8):
    """
    visualize HOG features

    returns
        None

    params
        target  : target image
        ihog    : integral HOG image
        cellsize: size of HOG feature to be visualized (default 8x8)

    """

    print ihog.shape
    ow, oh, _ = target.shape
    halfcell = cellsize / 2
    w, h = ow / cellsize, oh / cellsize
    norient = ihog.shape[2]
    mid = norient / 2

    for x in xrange(h - 1):
        for y in xrange(w - 1):
            px, py = x * cellsize, y * cellsize
            # feat = calc_hog_block(ihog, (px,py,max(px+cellsize, ow-1),max(py+cellsize, oh-1)))
            feat = calc_hog_block(ihog, (px, py, px + cellsize, py + cellsize))
            px += halfcell
            py += halfcell

            # L1-norm, nice for visualization
            mag = np.sum(feat)
            maxv = np.max(feat)
            if mag > 1e-3:
                nfeat = feat / maxv
                N = norient
                fdraw = []
                for i in xrange(N):
                    angmax = nfeat.argmax()
                    valmax = nfeat[angmax]
                    x1 = int(round(valmax * halfcell * np.sin((angmax - mid) * np.pi / mid)))
                    y1 = int(round(valmax * halfcell * np.cos((angmax - mid) * np.pi / mid)))
                    gv = int(round(255 * feat[angmax] / mag))

                    # don't draw if less than a threshold
                    if gv < 30:
                        break
                    fdraw.insert(0, (x1, y1, gv))
                    nfeat[angmax] = 0.

                # draw from smallest to highest gradient magnitude
                for i in xrange(len(fdraw)):
                    x1, y1, gv = fdraw[i]
                    cv2.line(target, (px - x1, py + y1), (px + x1, py - y1), (gv, gv, gv), 1, 8)
            else:
                # don't draw if there's no reponse
                pass
im = cv2.imread('Train Data\A9.jpg')

#image for visualization
vhog = np.zeros((im.shape), np.uint8)


hog = calc_hog(im)

draw_hog(vhog, hog, 8)


cv2.imshow("hog", vhog)


#clear for reuse
#vhog.set(0)

draw_hog(vhog, hog, 16)
cv2.imshow('lo', vhog)



key = cv2.waitKey(0)