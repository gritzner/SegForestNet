use num::{Integer, ToPrimitive};
use crate::utils::*;

//PYTHON_EXPORT u8,u16,i32; u8,u16,i32
pub fn accumulate_pixel_statistics<T: Copy + Integer + ToPrimitive, U: Copy + Integer + ToPrimitive>(
    mut accum_buffer: NdArray<u64>, mut accum_buffer_depth: NdArray<f64>,
    mut class_counts: NdArray<u64>, image: NdArray<T>, gt: NdArray<U>,
    depth: NdArray<f32>) {
        
    for y in 0..image.shape[0] {
        for x in 0..image.shape[1] {
            for c in 0..image.shape[2] {
                let val: T = image[[y, x, c]];
                let val = val.to_u64().unwrap();
                accum_buffer[[c,0]] += val;
                accum_buffer[[c,1]] += val * val;
            }
            
            let val = depth[[y, x]] as f64;
            accum_buffer_depth[0] += val;
            accum_buffer_depth[1] += val * val;

            let class: U = gt[[y,x]];
            let class = class.to_i32().unwrap();
            class_counts[class] += 1;            
        }
    }
}

//PYTHON_EXPORT u8,u16,i32
pub fn accumulate_pixel_statistics_sparse<T: Copy + Integer + ToPrimitive>(
    mut accum_buffer: NdArray<u64>, mut accum_buffer_depth: NdArray<f64>, mut class_counts: NdArray<u64>,
    image: NdArray<T>, instances: NdArray<u64>, masks: NdArray<u8>, depth: NdArray<f32>) {
    
    for y in 0..image.shape[0] {
        for x in 0..image.shape[1] {
            for c in 0..image.shape[2] {
                let val: T = image[[y, x, c]];
                let val = val.to_u64().unwrap();
                accum_buffer[[c,0]] += val;
                accum_buffer[[c,1]] += val * val;
            }
            
            let val = depth[[y, x]] as f64;
            accum_buffer_depth[0] += val;
            accum_buffer_depth[1] += val * val;
        }
    }
     
    class_counts[0] += (image.shape[0] * image.shape[1]) as u64;
    for i in 0..instances.shape[0] {
        let class = instances[[i, 0]] as i32;

        let height = instances[[i, 2]] - instances[[i, 1]];
        let width = instances[[i, 4]] - instances[[i, 3]];
        
        let begin = instances[[i, 5]] as i32;
        let end = begin + (height * width) as i32;
        
        for j in begin..end {
            if masks[j] != 0 {
                class_counts[class] += 1;
                class_counts[0] -= 1_u64;
            }
        }
    }
}
