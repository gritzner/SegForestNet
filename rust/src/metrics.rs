use crate::utils::*;
use std::collections::BTreeMap;

//PYTHON_EXPORT
pub fn add_to_conf_mat(mut conf_mat: NdArrayMut<u64>, yt: NdArray<i32>, yp: NdArray<i64>) {
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
pub fn match_segments(
    yt: NdArray<i32>, yt_inst: NdArray<i32>, yp_inst: NdArray<i32>, mut yt_ids: NdArrayMut<i32>,
    mut yt_ious: NdArrayMut<f32>, mut yp_ids: NdArrayMut<i32>, instance_class: i32, ignored_class: i32) {
        
    let mut yt_map = BTreeMap::new();
    for i in 0..yt_ids.shape[0] {
        yt_map.insert(yt_ids[[i, 0]], BTreeMap::new());
        yt_ids[[i, 1]] = 0;
    }
        
    let mut yp_map = BTreeMap::new();
    for i in 0..yp_ids.shape[0] {
        yp_map.insert(yp_ids[[i, 0]], vec![0, 0]);
        yp_ids[[i, 1]] = 0;
    }
        
    for y in 0..yt.shape[0] {
        for x in 0..yt.shape[1] {
            let yt_class = yt[[y, x]];
            let yt_id = yt_inst[[y, x]];
            let yp_id = yp_inst[[y, x]];
            
            if let Some(histogram) = yt_map.get_mut(&yt_id) {
                let value = (*histogram).entry(yp_id).or_insert(0);
                *value += 1;
            }
            
            if let Some(yp_size) = yp_map.get_mut(&yp_id) {
                yp_size[0] += 1;
                if (yt_class == instance_class && yp_id == 0) || yt_class == ignored_class {
                    yp_size[1] += 1;
                }
            }
        }
    }
        
    for i in 0..yt_ids.shape[0] {
        let yt_id = yt_ids[[i, 0]];
        
        if let Some(histogram) = yt_map.get(&yt_id) {
            let sum = histogram.values().fold(0, |a,b| a+b);
            
            for (yp_id, overlap) in histogram.iter() {
                let f_overlap = *overlap as f32;
                
                if let Some(yp_size) = yp_map.get(yp_id) {
                    let iou = f_overlap / ((sum + yp_size[0] - overlap) as f32);
                    
                    if iou > 0.5_f32 {
                        yt_ids[[i, 1]] = *yp_id;
                        yt_ious[i] = iou;
                        
                        for j in 0..yp_ids.shape[0] {
                            if yp_ids[[j, 0]] == *yp_id {
                                yp_ids[[j, 1]] = yt_id;
                                break;
                            }
                        }
                        
                        break;
                    }
                }
            }
        }
    }
        
    for i in 0..yp_ids.shape[0] {
        if yp_ids[[i, 1]] != 0 {
            continue;
        }
        
        let yp_id = yp_ids[[i, 0]];        
        if let Some(yp_size) = yp_map.get(&yp_id) {
            if ((yp_size[1] as f32) / (yp_size[0] as f32)) > 0.5_f32 {
                yp_ids[[i, 1]] = -1;
            }
        }
    }
}
