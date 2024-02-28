use crate::utils::*;

//PYTHON_EXPORT
pub fn add_to_conf_mat(mut conf_mat: NdArray<u64>, yt: NdArray<i32>, yp: NdArray<i64>) {
    for i in 0..yt.shape[0] {
        for j in 0..yt.shape[1] {
            for k in 0..yt.shape[2] {
                let true_class = yt[[i,j,k]];
                let predicted_class = yp[[i,j,k]] as i32;
                conf_mat[[true_class,predicted_class]] += 1;
            }
        }
    }
}

//PYTHON_EXPORT
pub fn prepare_per_pixel_entropy(mut p: NdArray<f64>, threshold: f64) {
    for y in 0..p.shape[0] {
        for x in 0..p.shape[1] {
            for c in 0..p.shape[2] {
                let i = [y, x, c];
                let v = p[i];
                if v < threshold {
                    p[i] = 0.0;
                } else {
                    p[i] = -v * v.ln();
                }
            }
        }
    }
}
