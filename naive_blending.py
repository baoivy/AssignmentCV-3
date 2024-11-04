import numpy as np
from PIL import Image

def naive_copy(source, target, mask, offset=(0, 0)):
    y_offset, x_offset = offset
    h, w, _ = source.shape

    target[y_offset:y_offset + h, x_offset:x_offset + w, :] = (
        target[y_offset:y_offset + h, x_offset:x_offset + w, :] * (1 - mask[..., None]) +
        source * mask[..., None]
    )

    return target

def main():
    img_src = np.array(Image.open('img/dog.jpg'))
    img_target = np.array(Image.open('img/pic1.jpg'))
    
    mask = np.zeros(img_src.shape[:2], dtype=np.uint8)
    mask[0:150, 0:150] = 1  
    
    offset = (0, 66)
    img_blended = naive_copy(img_src, img_target, mask, offset)
    Image.fromarray(img_blended.astype(np.uint8)).save('naive_blended3.jpg')

if __name__ == "__main__":
    main()


  