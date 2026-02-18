import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = np.ascontiguousarray(mask_image)
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_image(image, title="Image", ax=None):
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
    else:
        ax = ax 
    ax.imshow(image)
    ax.set_title(title, fontsize=18)
    ax.axis('off')
    return ax

def save_masked_image(image, masks, output_path, title="Detected Cracks"):
    """
    Overlays masks on the image and saves the result specifically for crack detection visualization.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    if len(masks) > 0:
        # Sort masks by area
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        for mask in sorted_masks:
            show_mask(mask['segmentation'], plt.gca(), random_color=True)
    plt.title(title, fontsize=18)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
