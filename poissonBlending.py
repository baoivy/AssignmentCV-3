import numpy as np
from PIL import Image
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, csr_matrix
import scipy.sparse
import cv2


def solveLinearSystem(a, b, y_range, x_range):
    x = spsolve(a, b)
    x = x.reshape((y_range, x_range))
    x = np.clip(x, 0, 255)
    return x.astype(np.uint8)

def laplacian_matrix(n, m):
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A

def poissonBlending(img_src, img_target, mask, offset = None):
    img_h_target, img_w_target, num_col = img_target.shape  

    mask = mask.astype(bool)
    mask = mask[0:img_h_target, 0:img_w_target]    
    mask[mask != 0] = 1
    mask_flat = mask.flatten()

    offset_x, offset_y = offset  
    affine_matrix = (1, 0, offset_x, 0, 1, offset_y)

    tmp_img = Image.fromarray(img_src)
    tmp_img = tmp_img.transform(
        (img_w_target, img_h_target), 
        Image.AFFINE,                  
        affine_matrix,                 
        resample=Image.BICUBIC        
    )
    img_src = np.array(tmp_img)

    mat_A = laplacian_matrix(img_h_target, img_w_target)
    laplacian = mat_A.tocsc()
    for y in range(1, img_h_target - 1):
        for x in range(1, img_w_target - 1):
            if mask[y, x] == 0:
                k = x + y * img_w_target
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + img_w_target] = 0
                mat_A[k, k - img_w_target] = 0

    mat_A = mat_A.tocsc()
    img_blended = np.copy(img_target)
    for col in range(num_col):
        source_flat = img_src[:, :, col].flatten()
        target_flat = img_target[:, :, col].flatten()        
            
        mat_b = laplacian.dot(source_flat)

        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        x = solveLinearSystem(mat_A, mat_b, img_h_target, img_w_target)
        
        img_blended[:, :, col] = x
    
    return img_blended

def main():
    img_src = np.array(Image.open('img/cat_224x224.jpg').resize((224, 224)))
    print(img_src.shape)
    img_target = np.array(Image.open('img/pic2.jpg').resize((224, 224)))
    mask = np.zeros(img_src.shape[:2], dtype=np.uint8)
    mask[50:150, 0:150] = 1  
    offset = (50,66)
    img_blended = poissonBlending(img_src, img_target, mask, offset)
    Image.fromarray(img_blended.astype(np.uint8)).save('cat_pic2.jpg')

if __name__ == "__main__":
    main()