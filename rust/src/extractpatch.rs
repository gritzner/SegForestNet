use num::{Integer, ToPrimitive};
use crate::utils::*;
use crate::sparseimage::{convert_to_sparse_map, get_pixel_sparse};
use rand::SeedableRng;
use rand_xorshift::XorShiftRng;
use rand_distr::{Distribution, StandardNormal};

macro_rules! clip {
    ( $i: ident, $limit: expr ) => {
        let limit = 2 * $limit;
        let i = $i % limit;
        let $i = if i < $limit { i } else { limit - (i+1) };
    }
}

macro_rules! adjust_range {
    ( $i: ident, $j: ident, $f: ident, $val: expr, $limit: expr ) => {
        let f = $val.abs();
        let $f = f.fract();
        
        let $i = f as i32;
        let $j = $i + 1;
        
        clip!($i, $limit);        
        clip!($j, $limit);
    }
}

macro_rules! for_each_augmented_pixel {
    ( ($out_size: ident, $img_size: ident, $transform: expr, $flips: expr) -> ($y: ident, $x: ident), $sample_pos: ident, $bilerp_factors: ident $code: block ) => {
        let height_factor = 1. / (($out_size.0 - 1) as f32);
        let width_factor = 1. / (($out_size.1 - 1) as f32);
        
        for y in 0..$out_size.0 {
            let fy = ((2*y) as f32) * height_factor;
            let fy = fy - 1.;
            let $y = if $flips[1] == 0_u8 { y } else { $out_size.0 - (y + 1) };
            
            for x in 0..$out_size.1 {
                let fx = ((2*x) as f32) * width_factor;
                let fx = fx - 1.;
                let $x = if $flips[0] == 0_u8 { x } else { $out_size.1 - (x + 1) };
                
                let sample_pos: (f32, f32) = (
                    $transform[[0, 0]] * fx + $transform[[0, 1]] * fy + $transform[[0, 2]],
                    $transform[[1, 0]] * fx + $transform[[1, 1]] * fy + $transform[[1, 2]],                    
                );
                
                adjust_range!(fxi, fxii, fxf, sample_pos.0, $img_size.1);
                adjust_range!(fyi, fyii, fyf, sample_pos.1, $img_size.0);

                let $sample_pos = (
                    (fyi, fxi),
                    (fyi, fxii),
                    (fyii, fxi),
                    (fyii, fxii)
                );
                
                let $bilerp_factors = (
                    (1. - fyf) * (1. - fxf),
                    (1. - fyf) * fxf,
                    fyf * (1. - fxf),
                    fyf * fxf
                );
                
                $code
            }
        }
    }
}

fn ndvi<T: Copy + Integer + ToPrimitive>(img: &NdArray<T>, y: i32, x: i32, ir_index: i32, red_index: i32) -> f32 {
    let ir = img[[y, x, ir_index]];
    let red = img[[y, x, red_index]];
    let denom = ir + red;
    if denom.is_zero() {
        0.
    } else {
        (ir.to_f32().unwrap() - red.to_f32().unwrap()) / denom.to_f32().unwrap()
    }
}

//PYTHON_EXPORT u8,u16,i32; u8,u16,i32
pub fn extract_patch_bilinear_normalized<T: Copy + Integer + ToPrimitive, U: Copy + Integer + ToPrimitive>(
    mut out: NdArrayMut<f32>, img: NdArray<T>, gt: NdArray<U>, depth: NdArray<f32>,
    transform: NdArray<f32>, flips: NdArray<u8>, seed: u64, noise_magnitude: f32,
    channels: NdArray<i32>, ir_index: i32, red_index: i32,
    normalization_params: NdArray<f32>, num_classes: i32) {

    let out_size = (out.shape[1], out.shape[2]);
    let img_size = (img.shape[0], img.shape[1]);
    let mut rng = XorShiftRng::seed_from_u64(seed);
    let num_classes = 0.5_f32 * (num_classes as f32);
        
    for_each_augmented_pixel!(
        (out_size, img_size, transform, flips) -> (y, x), sample_pos, bilerp_factors {
            for c in 0..out.shape[0] {
                let mut val: f32 = 0.;
                
                if channels[c] == -1 {
                    val += bilerp_factors.0 * (gt[[(sample_pos.0).0, (sample_pos.0).1]] as U).to_f32().unwrap();
                    val += bilerp_factors.1 * (gt[[(sample_pos.1).0, (sample_pos.1).1]] as U).to_f32().unwrap();
                    val += bilerp_factors.2 * (gt[[(sample_pos.2).0, (sample_pos.2).1]] as U).to_f32().unwrap();
                    val += bilerp_factors.3 * (gt[[(sample_pos.3).0, (sample_pos.3).1]] as U).to_f32().unwrap();
                    out[[c, y, x]] = (val - num_classes) / num_classes;
                } else if channels[c] == -2 {
                    val += bilerp_factors.0 * depth[[(sample_pos.0).0, (sample_pos.0).1]];
                    val += bilerp_factors.1 * depth[[(sample_pos.1).0, (sample_pos.1).1]];
                    val += bilerp_factors.2 * depth[[(sample_pos.2).0, (sample_pos.2).1]];
                    val += bilerp_factors.3 * depth[[(sample_pos.3).0, (sample_pos.3).1]];
                    val = (val - normalization_params[[img.shape[2], 0]]) / normalization_params[[img.shape[2], 1]];
                } else if channels[c] == -3 {
                    val += bilerp_factors.0 * ndvi(&img, (sample_pos.0).0, (sample_pos.0).1, ir_index, red_index);
                    val += bilerp_factors.1 * ndvi(&img, (sample_pos.1).0, (sample_pos.1).1, ir_index, red_index);
                    val += bilerp_factors.2 * ndvi(&img, (sample_pos.2).0, (sample_pos.2).1, ir_index, red_index);
                    val += bilerp_factors.3 * ndvi(&img, (sample_pos.3).0, (sample_pos.3).1, ir_index, red_index);
                } else {
                    let c_in = channels[c];
                    val += bilerp_factors.0 * img[[(sample_pos.0).0, (sample_pos.0).1, c_in]].to_f32().unwrap();
                    val += bilerp_factors.1 * img[[(sample_pos.1).0, (sample_pos.1).1, c_in]].to_f32().unwrap();
                    val += bilerp_factors.2 * img[[(sample_pos.2).0, (sample_pos.2).1, c_in]].to_f32().unwrap();
                    val += bilerp_factors.3 * img[[(sample_pos.3).0, (sample_pos.3).1, c_in]].to_f32().unwrap();
                    val = (val - normalization_params[[c_in, 0]]) / normalization_params[[c_in, 1]];
                }
                
                if channels[c] != -1 {
                    let noise: f32 = StandardNormal.sample(&mut rng);
                    out[[c, y, x]] = val + (noise * noise_magnitude);
                }
            }
        }
    );
}

//PYTHON_EXPORT u8,u16,i32
pub fn extract_patch_bilinear_normalized_sparse<T: Copy + Integer + ToPrimitive>(
    mut out: NdArrayMut<f32>, img: NdArray<T>, instances: NdArray<u64>, rows: NdArray<i32>,
    cols: NdArray<i32>, masks: NdArray<u8>, depth: NdArray<f32>, transform: NdArray<f32>,
    flips: NdArray<u8>, seed: u64, noise_magnitude: f32, channels: NdArray<i32>,
    ir_index: i32, red_index: i32, normalization_params: NdArray<f32>, num_classes: i32) {

    let out_size = (out.shape[1], out.shape[2]);
    let img_size = (img.shape[0], img.shape[1]);
    let mut rng = XorShiftRng::seed_from_u64(seed);
    let rows = convert_to_sparse_map(&rows);
    let num_classes = 0.5_f32 * (num_classes as f32);
        
    for_each_augmented_pixel!(
        (out_size, img_size, transform, flips) -> (y, x), sample_pos, bilerp_factors {
            for c in 0..out.shape[0] {
                let mut val: f32 = 0.;
                
                if channels[c] == -1 {
                    val += bilerp_factors.0 * get_pixel_sparse((sample_pos.0).0, (sample_pos.0).1, &instances, &rows[&(sample_pos.0).0], &cols, &masks, true) as f32;
                    val += bilerp_factors.1 * get_pixel_sparse((sample_pos.1).0, (sample_pos.1).1, &instances, &rows[&(sample_pos.1).0], &cols, &masks, true) as f32;
                    val += bilerp_factors.2 * get_pixel_sparse((sample_pos.2).0, (sample_pos.2).1, &instances, &rows[&(sample_pos.2).0], &cols, &masks, true) as f32;
                    val += bilerp_factors.3 * get_pixel_sparse((sample_pos.3).0, (sample_pos.3).1, &instances, &rows[&(sample_pos.3).0], &cols, &masks, true) as f32;
                    out[[c, y, x]] = (val - num_classes) / num_classes;
                } else if channels[c] == -2 {
                    val += bilerp_factors.0 * depth[[(sample_pos.0).0, (sample_pos.0).1]];
                    val += bilerp_factors.1 * depth[[(sample_pos.1).0, (sample_pos.1).1]];
                    val += bilerp_factors.2 * depth[[(sample_pos.2).0, (sample_pos.2).1]];
                    val += bilerp_factors.3 * depth[[(sample_pos.3).0, (sample_pos.3).1]];
                    val = (val - normalization_params[[img.shape[2], 0]]) / normalization_params[[img.shape[2], 1]];
                } else if channels[c] == -3 {
                    val += bilerp_factors.0 * ndvi(&img, (sample_pos.0).0, (sample_pos.0).1, ir_index, red_index);
                    val += bilerp_factors.1 * ndvi(&img, (sample_pos.1).0, (sample_pos.1).1, ir_index, red_index);
                    val += bilerp_factors.2 * ndvi(&img, (sample_pos.2).0, (sample_pos.2).1, ir_index, red_index);
                    val += bilerp_factors.3 * ndvi(&img, (sample_pos.3).0, (sample_pos.3).1, ir_index, red_index);
                } else {
                    let c_in = channels[c];
                    val += bilerp_factors.0 * img[[(sample_pos.0).0, (sample_pos.0).1, c_in]].to_f32().unwrap();
                    val += bilerp_factors.1 * img[[(sample_pos.1).0, (sample_pos.1).1, c_in]].to_f32().unwrap();
                    val += bilerp_factors.2 * img[[(sample_pos.2).0, (sample_pos.2).1, c_in]].to_f32().unwrap();
                    val += bilerp_factors.3 * img[[(sample_pos.3).0, (sample_pos.3).1, c_in]].to_f32().unwrap();
                    val = (val - normalization_params[[c_in, 0]]) / normalization_params[[c_in, 1]];
                }
                
                if channels[c] != -1 {
                    let noise: f32 = StandardNormal.sample(&mut rng);
                    out[[c, y, x]] = val + (noise * noise_magnitude);
                }
            }
        }
    );
}

//PYTHON_EXPORT u8,u16,i32; u8,u16,i32
pub fn extract_patch_bilinear<T: Copy + Integer + ToPrimitive, U: Copy + Integer + ToPrimitive>(
    mut out: NdArrayMut<u8>, img: NdArray<T>, gt: NdArray<U>,
    depth: NdArray<f32>, transform: NdArray<f32>, flips: NdArray<u8>,
    channels: NdArray<i32>, ir_index: i32, red_index: i32,
    depth_range: NdArray<f32>) {

    let out_size = (out.shape[1], out.shape[2]);
    let img_size = (img.shape[0], img.shape[1]);
    let depth_range = (
        depth_range[0],
        255.0_f32 / (depth_range[1] - depth_range[0])
    );
        
    for_each_augmented_pixel!(
        (out_size, img_size, transform, flips) -> (y, x), sample_pos, bilerp_factors {
            for c in 0..out.shape[0] {
                let mut val: f32 = 0.;
                
                if channels[c] == -1 {
                    val += bilerp_factors.0 * (gt[[(sample_pos.0).0, (sample_pos.0).1]] as U).to_f32().unwrap();
                    val += bilerp_factors.1 * (gt[[(sample_pos.1).0, (sample_pos.1).1]] as U).to_f32().unwrap();
                    val += bilerp_factors.2 * (gt[[(sample_pos.2).0, (sample_pos.2).1]] as U).to_f32().unwrap();
                    val += bilerp_factors.3 * (gt[[(sample_pos.3).0, (sample_pos.3).1]] as U).to_f32().unwrap();
                } else if channels[c] == -2 {
                    val += bilerp_factors.0 * depth[[(sample_pos.0).0, (sample_pos.0).1]];
                    val += bilerp_factors.1 * depth[[(sample_pos.1).0, (sample_pos.1).1]];
                    val += bilerp_factors.2 * depth[[(sample_pos.2).0, (sample_pos.2).1]];
                    val += bilerp_factors.3 * depth[[(sample_pos.3).0, (sample_pos.3).1]];
                    val = (val - depth_range.0) * depth_range.1;
                } else if channels[c] == -3 {
                    val += bilerp_factors.0 * ndvi(&img, (sample_pos.0).0, (sample_pos.0).1, ir_index, red_index);
                    val += bilerp_factors.1 * ndvi(&img, (sample_pos.1).0, (sample_pos.1).1, ir_index, red_index);
                    val += bilerp_factors.2 * ndvi(&img, (sample_pos.2).0, (sample_pos.2).1, ir_index, red_index);
                    val += bilerp_factors.3 * ndvi(&img, (sample_pos.3).0, (sample_pos.3).1, ir_index, red_index);
                    val = 127.5 * (val + 1.);
                } else {
                    let c_in = channels[c];
                    val += bilerp_factors.0 * img[[(sample_pos.0).0, (sample_pos.0).1, c_in]].to_f32().unwrap();
                    val += bilerp_factors.1 * img[[(sample_pos.1).0, (sample_pos.1).1, c_in]].to_f32().unwrap();
                    val += bilerp_factors.2 * img[[(sample_pos.2).0, (sample_pos.2).1, c_in]].to_f32().unwrap();
                    val += bilerp_factors.3 * img[[(sample_pos.3).0, (sample_pos.3).1, c_in]].to_f32().unwrap();
                }
                
                out[[c, y, x]] = val.round() as u8;
            }
        }
    );
}

//PYTHON_EXPORT u8,u16,i32
pub fn extract_patch_bilinear_sparse<T: Copy + Integer + ToPrimitive>(
    mut out: NdArrayMut<u8>, img: NdArray<T>, instances: NdArray<u64>, rows: NdArray<i32>,
    cols: NdArray<i32>, masks: NdArray<u8>, depth: NdArray<f32>, transform: NdArray<f32>,
    flips: NdArray<u8>, channels: NdArray<i32>, ir_index: i32, red_index: i32,
    depth_range: NdArray<f32>) {

    let out_size = (out.shape[1], out.shape[2]);
    let img_size = (img.shape[0], img.shape[1]);
    let rows = convert_to_sparse_map(&rows);
    let depth_range = (
        depth_range[0],
        255.0_f32 / (depth_range[1] - depth_range[0])
    );
        
    for_each_augmented_pixel!(
        (out_size, img_size, transform, flips) -> (y, x), sample_pos, bilerp_factors {
            for c in 0..out.shape[0] {
                let mut val: f32 = 0.;
                
                if channels[c] == -1 {
                    val += bilerp_factors.0 * get_pixel_sparse((sample_pos.0).0, (sample_pos.0).1, &instances, &rows[&(sample_pos.0).0], &cols, &masks, true) as f32;
                    val += bilerp_factors.1 * get_pixel_sparse((sample_pos.1).0, (sample_pos.1).1, &instances, &rows[&(sample_pos.1).0], &cols, &masks, true) as f32;
                    val += bilerp_factors.2 * get_pixel_sparse((sample_pos.2).0, (sample_pos.2).1, &instances, &rows[&(sample_pos.2).0], &cols, &masks, true) as f32;
                    val += bilerp_factors.3 * get_pixel_sparse((sample_pos.3).0, (sample_pos.3).1, &instances, &rows[&(sample_pos.3).0], &cols, &masks, true) as f32;
                } else if channels[c] == -2 {
                    val += bilerp_factors.0 * depth[[(sample_pos.0).0, (sample_pos.0).1]];
                    val += bilerp_factors.1 * depth[[(sample_pos.1).0, (sample_pos.1).1]];
                    val += bilerp_factors.2 * depth[[(sample_pos.2).0, (sample_pos.2).1]];
                    val += bilerp_factors.3 * depth[[(sample_pos.3).0, (sample_pos.3).1]];
                    val = (val - depth_range.0) * depth_range.1;
                } else if channels[c] == -3 {
                    val += bilerp_factors.0 * ndvi(&img, (sample_pos.0).0, (sample_pos.0).1, ir_index, red_index);
                    val += bilerp_factors.1 * ndvi(&img, (sample_pos.1).0, (sample_pos.1).1, ir_index, red_index);
                    val += bilerp_factors.2 * ndvi(&img, (sample_pos.2).0, (sample_pos.2).1, ir_index, red_index);
                    val += bilerp_factors.3 * ndvi(&img, (sample_pos.3).0, (sample_pos.3).1, ir_index, red_index);
                    val = 127.5 * (val + 1.);
                } else {
                    let c_in = channels[c];
                    val += bilerp_factors.0 * img[[(sample_pos.0).0, (sample_pos.0).1, c_in]].to_f32().unwrap();
                    val += bilerp_factors.1 * img[[(sample_pos.1).0, (sample_pos.1).1, c_in]].to_f32().unwrap();
                    val += bilerp_factors.2 * img[[(sample_pos.2).0, (sample_pos.2).1, c_in]].to_f32().unwrap();
                    val += bilerp_factors.3 * img[[(sample_pos.3).0, (sample_pos.3).1, c_in]].to_f32().unwrap();
                }
                
                out[[c, y, x]] = val.round() as u8;
            }
        }
    );
}

//PYTHON_EXPORT u8,u16,i32
pub fn extract_patch_nearest<T: Copy + Integer + ToPrimitive>(
    mut out: NdArrayMut<i32>, img: NdArray<T>,
    transform: NdArray<f32>, flips: NdArray<u8>) {
    
    let out_size = (out.shape[0], out.shape[1]);
    let img_size = (img.shape[0], img.shape[1]);
    let mut classes = [T::zero(), T::zero(), T::zero(), T::zero()];
    let mut votes = [0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32];
       
    for_each_augmented_pixel!(
        (out_size, img_size, transform, flips) -> (y, x), sample_pos, bilerp_factors {
            classes[0] = img[[(sample_pos.0).0, (sample_pos.0).1]];
            votes[0] = bilerp_factors.0;
            
            classes[1] = img[[(sample_pos.1).0, (sample_pos.1).1]];
            votes[1] = bilerp_factors.1;
            
            classes[2] = img[[(sample_pos.2).0, (sample_pos.2).1]];
            votes[2] = bilerp_factors.2;
            
            classes[3] = img[[(sample_pos.3).0, (sample_pos.3).1]];
            votes[3] = bilerp_factors.3;
            
            for c in 0..3 {
                for c2 in (c+1)..4 {
                    if classes[c] == classes[c2] {
                        votes[c] += votes[c2];
                        votes[c2] = 0.0_f32;
                    }
                }
            }
            
            let mut winner = 0;
            for c in 1..4 {
                if votes[c] > votes[winner] {
                    winner = c;
                }
            }
        
            out[[y, x]] = classes[winner].to_i32().unwrap();
        }
    );
}

//PYTHON_EXPORT
pub fn extract_patch_nearest_sparse(
    mut out: NdArrayMut<i32>, instances: NdArray<u64>, rows: NdArray<i32>, cols: NdArray<i32>,
    masks: NdArray<u8>, return_classes: bool, transform: NdArray<f32>, flips: NdArray<u8>) {
    
    let out_size = (out.shape[0], out.shape[1]);
    let img_size = (rows.shape[0], cols.shape[0]);
    let rows = convert_to_sparse_map(&rows);
    let mut classes = [0_i32, 0_i32, 0_i32, 0_i32];
    let mut votes = [0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32];
       
    for_each_augmented_pixel!(
        (out_size, img_size, transform, flips) -> (y, x), sample_pos, bilerp_factors {
            classes[0] = get_pixel_sparse((sample_pos.0).0, (sample_pos.0).1, &instances, &rows[&(sample_pos.0).0], &cols, &masks, return_classes);
            votes[0] = bilerp_factors.0;
            
            classes[1] = get_pixel_sparse((sample_pos.1).0, (sample_pos.1).1, &instances, &rows[&(sample_pos.1).0], &cols, &masks, return_classes);
            votes[1] = bilerp_factors.1;
            
            classes[2] = get_pixel_sparse((sample_pos.2).0, (sample_pos.2).1, &instances, &rows[&(sample_pos.2).0], &cols, &masks, return_classes);
            votes[2] = bilerp_factors.2;
            
            classes[3] = get_pixel_sparse((sample_pos.3).0, (sample_pos.3).1, &instances, &rows[&(sample_pos.3).0], &cols, &masks, return_classes);
            votes[3] = bilerp_factors.3;
            
            for c in 0..3 {
                for c2 in (c+1)..4 {
                    if classes[c] == classes[c2] {
                        votes[c] += votes[c2];
                        votes[c2] = 0.0_f32;
                    }
                }
            }
            
            let mut winner = 0;
            for c in 1..4 {
                if votes[c] > votes[winner] {
                    winner = c;
                }
            }
        
            out[[y, x]] = classes[winner];
        }
    );
}

macro_rules! for_each_pixel {
    ( ($height: expr, $width: expr, $offsets: ident) -> ($y: ident, $x: ident), $sample_pos: ident $code: block ) => {
        for $y in 0..$height {
            let y_offset = $y + $offsets[0];
            
            for $x in 0..$width {
                let $sample_pos = (
                    y_offset,
                    $x + $offsets[1]
                );
                
                $code
            }
        }
    }
}

//PYTHON_EXPORT u8,u16,i32; u8,u16,i32
pub fn get_patch_normalized<T: Copy + Integer + ToPrimitive, U: Copy + Integer + ToPrimitive>(
    mut out: NdArrayMut<f32>, img: NdArray<T>, gt: NdArray<U>, depth: NdArray<f32>,
    offsets: NdArray<i32>, channels: NdArray<i32>, ir_index: i32, red_index: i32,
    normalization_params: NdArray<f32>, num_classes: i32) {

    let num_classes = 0.5_f32 * (num_classes as f32);
    
    for_each_pixel!(
        (out.shape[1], out.shape[2], offsets) -> (y, x), sample_pos {
            for c in 0..out.shape[0] {
                let mut val: f32;
                
                if channels[c] == -1 {
                    val = (gt[[sample_pos.0, sample_pos.1]] as U).to_f32().unwrap();
                    val = (val - num_classes) / num_classes;
                } else if channels[c] == -2 {
                    val = depth[[sample_pos.0, sample_pos.1]];
                    val = (val - normalization_params[[img.shape[2], 0]]) / normalization_params[[img.shape[2], 1]];
                } else if channels[c] == -3 {
                    val = ndvi(&img, sample_pos.0, sample_pos.1, ir_index, red_index);
                } else {
                    let c_in = channels[c];
                    val = img[[sample_pos.0, sample_pos.1, c_in]].to_f32().unwrap();
                    val = (val - normalization_params[[c_in, 0]]) / normalization_params[[c_in, 1]];
                }
                
                out[[c, y, x]] = val;
            }
        }
    );
}

//PYTHON_EXPORT u8,u16,i32
pub fn get_patch_normalized_sparse<T: Copy + Integer + ToPrimitive>(
    mut out: NdArrayMut<f32>, img: NdArray<T>, instances: NdArray<u64>, rows: NdArray<i32>,
    cols: NdArray<i32>, masks: NdArray<u8>,depth: NdArray<f32>, offsets: NdArray<i32>,
    channels: NdArray<i32>, ir_index: i32, red_index: i32, normalization_params: NdArray<f32>,
    num_classes: i32) {

    let rows = convert_to_sparse_map(&rows);        
    let num_classes = 0.5_f32 * (num_classes as f32);
    
    for_each_pixel!(
        (out.shape[1], out.shape[2], offsets) -> (y, x), sample_pos {
            for c in 0..out.shape[0] {
                let mut val: f32;
                
                if channels[c] == -1 {
                    val = get_pixel_sparse(sample_pos.0, sample_pos.1, &instances, &rows[&sample_pos.0], &cols, &masks, true) as f32;
                    val = (val - num_classes) / num_classes;
                } else if channels[c] == -2 {
                    val = depth[[sample_pos.0, sample_pos.1]];
                    val = (val - normalization_params[[img.shape[2], 0]]) / normalization_params[[img.shape[2], 1]];
                } else if channels[c] == -3 {
                    val = ndvi(&img, sample_pos.0, sample_pos.1, ir_index, red_index);
                } else {
                    let c_in = channels[c];
                    val = img[[sample_pos.0, sample_pos.1, c_in]].to_f32().unwrap();
                    val = (val - normalization_params[[c_in, 0]]) / normalization_params[[c_in, 1]];
                }
                
                out[[c, y, x]] = val;
            }
        }
    );
}

//PYTHON_EXPORT u8,u16,i32; u8,u16,i32
pub fn get_patch<T: Copy + Integer + ToPrimitive, U: Copy + Integer + ToPrimitive>(
    mut out: NdArrayMut<u8>, img: NdArray<T>, gt: NdArray<U>, depth: NdArray<f32>,
    offsets: NdArray<i32>, channels: NdArray<i32>, ir_index: i32, red_index: i32,
    depth_range: NdArray<f32>) {

    let depth_range = (
        depth_range[0],
        255.0_f32 / (depth_range[1] - depth_range[0])
    );
            
    for_each_pixel!(
        (out.shape[1], out.shape[2], offsets) -> (y, x), sample_pos {
            for c in 0..out.shape[0] {
                let mut val: f32;
                
                if channels[c] == -1 {
                    val = (gt[[sample_pos.0, sample_pos.1]] as U).to_f32().unwrap();
                } else if channels[c] == -2 {
                    val = depth[[sample_pos.0, sample_pos.1]];
                    val = (val - depth_range.0) * depth_range.1;
                } else if channels[c] == -3 {
                    val = ndvi(&img, sample_pos.0, sample_pos.1, ir_index, red_index);
                    val = 127.5 * (val + 1.);
                } else {
                    val = img[[sample_pos.0, sample_pos.1, channels[c]]].to_f32().unwrap();
                }
                
                out[[c, y, x]] = val.round() as u8;
            }
        }
    );
}

//PYTHON_EXPORT u8,u16,i32
pub fn get_patch_sparse<T: Copy + Integer + ToPrimitive>(
    mut out: NdArrayMut<u8>, img: NdArray<T>, instances: NdArray<u64>, rows: NdArray<i32>, 
    cols: NdArray<i32>, masks: NdArray<u8>, depth: NdArray<f32>, offsets: NdArray<i32>,
    channels: NdArray<i32>, ir_index: i32, red_index: i32, depth_range: NdArray<f32>) {

    let rows = convert_to_sparse_map(&rows);        
    let depth_range = (
        depth_range[0],
        255.0_f32 / (depth_range[1] - depth_range[0])
    );
            
    for_each_pixel!(
        (out.shape[1], out.shape[2], offsets) -> (y, x), sample_pos {
            for c in 0..out.shape[0] {
                let mut val: f32;
                
                if channels[c] == -1 {
                    val = get_pixel_sparse(sample_pos.0, sample_pos.1, &instances, &rows[&sample_pos.0], &cols, &masks, true) as f32;
                } else if channels[c] == -2 {
                    val = depth[[sample_pos.0, sample_pos.1]];
                    val = (val - depth_range.0) * depth_range.1;
                } else if channels[c] == -3 {
                    val = ndvi(&img, sample_pos.0, sample_pos.1, ir_index, red_index);
                    val = 127.5 * (val + 1.);
                } else {
                    val = img[[sample_pos.0, sample_pos.1, channels[c]]].to_f32().unwrap();
                }
                
                out[[c, y, x]] = val.round() as u8;
            }
        }
    );
}

//PYTHON_EXPORT u8,u16,i32
pub fn get_gt_patch<T: Copy + Integer + ToPrimitive>(
    mut out: NdArrayMut<i32>, img: NdArray<T>, offsets: NdArray<i32>) {
        
    for_each_pixel!(
        (out.shape[0], out.shape[1], offsets) -> (y, x), sample_pos {
            out[[y, x]] = (img[[sample_pos.0, sample_pos.1]] as T).to_i32().unwrap();
        }
    );
}

//PYTHON_EXPORT
pub fn get_gt_patch_sparse(mut out: NdArrayMut<i32>, instances: NdArray<u64>, rows: NdArray<i32>, cols: NdArray<i32>, masks: NdArray<u8>, return_classes: bool, offsets: NdArray<i32>) {
        
    let rows = convert_to_sparse_map(&rows);
    
    for_each_pixel!(
        (out.shape[0], out.shape[1], offsets) -> (y, x), sample_pos {
            out[[y, x]] = get_pixel_sparse(sample_pos.0, sample_pos.1, &instances, &rows[&sample_pos.0], &cols, &masks, return_classes);
        }
    );
}
