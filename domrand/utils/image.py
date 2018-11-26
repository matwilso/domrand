import numpy as np
import matplotlib.pyplot as plt
import cv2
#cv2.ocl.setUseOpenCL(False)
from domrand.utils.constants import TOX, TOY, GRID_SPACING, TWX, TWY, TBS, TBE, TBSX, TBSY, TBEX, TBEY
from domrand.utils.general import softmax, bin_to_xyz_np

# IMAGE UTILS
def display_image(cam_img, real_img_path='./data/real/3-3.jpg', mode='preproc'):
    """matplotlib show image"""
    if real_img_path is not None:
        real_img = plt.imread(real_img_path)
    if mode == 'preproc':
        real_img = preproc_image(real_img)
        cam_img = preproc_image(cam_img)

    fig = plt.figure()

    cp = lambda x,y: np.clip(x+y, 0, 255).astype(int)
    subp = fig.add_subplot(1,2,1)
    imgplot = plt.imshow(cp(cam_img[...,:], 0.0*255))
    subp.set_title('Sim')

    subp = fig.add_subplot(1,2,2)
    imgplot = plt.imshow(real_img[...,:])
    subp.set_title('Real')

    #fig.text(0, 0, label)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    #mng.window.state('zoomed') #works fine on Windows!
    plt.show()
    
def preproc_image(img, width=224, height=224, dtype=np.uint8):
    """Resize image to the given shape"""
    # reference: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py 
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img.astype(dtype)

TABLE_LINES = np.array([
    [TOX, TOX, TOY, TOY-TWY],
    [TOX, TOX-TWX, TOY, TOY],
    [TOX-TWX, TOX-TWX, TOY, TOY-TWY],
    [TOX, TOX-TWX, TOY-TWY, TOY-TWY]
])[:,::-1]

def _make_xyz_plot(image, pred, label):
    """
    Plot the image, and a visualization of the prediction compared to the ground truth when using XYZ coords
    """
    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.imshow(image, aspect='auto')
    ax1.set_title('image')

    ax2.set_title('plot')
    ax2.axis('equal')
    ax2.plot([TABLE_LINES[:,0], TABLE_LINES[:,1]], [TABLE_LINES[:,2], TABLE_LINES[:,3]], color='black')
    ax2.plot(*pred[:2][::-1], 'o', label='pred: {}'.format(pred))
    if label is not None:
        ax2.plot(*label[:2][::-1], 'x', label='label: {}'.format(label))
        euc = np.linalg.norm(label[:2] - pred[:2])
        ax2.text(0.0, -0.25, 'euc: {0:0.2f}'.format(euc), horizontalalignment='center')

    ax2.legend()
    plt.gca().invert_xaxis()

    #asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    #ax2.set_aspect(asp)
    #plt.gca().invert_yaxis()
    #ax2.text(0, 0, 'test')

    fig.canvas.draw()
    X = np.array(fig.canvas.renderer._renderer)
    fig.clf()
    plt.close()
    return X[...,:3]

def _make_binned_plot(image, pred, sparse_label):
    """
    Plot the image, and a visualization of the prediction compared to the ground truth when using binned coords
    """
    pred = softmax(pred, axis=1)
    combined_prob = np.outer(pred[0], pred[1])
    bins = pred.shape[-1]

    ## weighted sum center
    pred_cx = np.sum(pred[0] * np.linspace(TBS[0], TBE[0], pred.shape[-1]))
    pred_cy = np.sum(pred[1] * np.linspace(TBS[1], TBE[1], pred.shape[-1]))
    pred_cz = np.sum(pred[2] * np.linspace(TBS[2], TBE[2], pred.shape[-1]))
    pred_c = np.array([pred_cx, pred_cy, pred_cz], np.float32)

    # argmax center
    pred_a = bin_to_xyz_np(np.argmax(pred, 1), bins)
    label = bin_to_xyz_np(sparse_label, bins)

    fig, (ax1, ax2) = plt.subplots(1,2)

    ax1.imshow(image)#, aspect='auto')
    ax1.set_title('image')

    ax2.set_title('plot')
    ax2.axis('equal')
    ax2.plot([TABLE_LINES[:,0], TABLE_LINES[:,1]], [TABLE_LINES[:,2], TABLE_LINES[:,3]], color='black')
    ax2.plot(*pred_a[:2][::-1], 'o', color='C0', label='pred: {}'.format(pred_a))
    #ax2.plot(*pred_c[:2][::-1], '^', color='C9', alpha=0.5, label='pred softmax: {}'.format(pred_c))

    if label is not None:
        ax2.plot(*label[:2][::-1], 'x', color='C3', label='label: {}'.format(label))
        euc = np.linalg.norm(label[:2] - pred_a[:2])
        ax2.text(0.0, -0.25, 'euc: {0:0.2f}'.format(euc), horizontalalignment='center')

    ax2.legend()
    ax2.invert_xaxis()

    fig.canvas.draw()
    X = np.array(fig.canvas.renderer._renderer)
    fig.clf()
    plt.close()
    return X[...,:3]



def make_pred_plot(image, pred, label=None, mode=None):
    """
    takes about 0.15s per image, so kinda expensive
    """
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    if mode == 'xyz':
        return _make_xyz_plot(image, pred, label)
    elif mode == 'binned':
        return _make_binned_plot(image, pred, label)
    else:
        raise Exception('Need to choose mode: ("xyz" or "binned")')


