use crate::utils::*;
use std::collections::BTreeMap;

//PYTHON_EXPORT
pub fn rgb2ids(mut gt_ids: NdArray<u8>, gt_rgb: NdArray<u8>, mut inst_ids: NdArray<i32>, inst_rgb: NdArray<u8>, class_map_array: NdArray<u8>) -> usize {
    let mut class_map = BTreeMap::new();
    for i in 0..class_map_array.shape[0] {
        class_map.insert((class_map_array[[i,0]], class_map_array[[i,1]], class_map_array[[i,2]]), i as u8);
    }
    
    let mut instance_map = BTreeMap::new();
    
    for y in 0..gt_rgb.shape[0] {
        for x in 0..gt_rgb.shape[1] {
            let class_id = class_map[&(gt_rgb[[y,x,0]], gt_rgb[[y,x,1]], gt_rgb[[y,x,2]])];
            gt_ids[[y, x]] = class_id;
            
            let mut instance_id = 0;
            if class_id > 0 {
                let new_id = (instance_map.len() + 1) as i32;
                let instance_rgb = (inst_rgb[[y,x,0]], inst_rgb[[y,x,1]], inst_rgb[[y,x,2]]);
                let instance = instance_map.entry(instance_rgb).or_insert(new_id);
                instance_id = *instance;
            }
            inst_ids[[y, x]] = instance_id;
        }
    }
    
    instance_map.len()
}

//PYTHON_EXPORT
pub fn extract_instances(mut instances: NdArray<u64>, gt: NdArray<u8>, inst_img: NdArray<i32>) {
    for i in 0..instances.shape[0] {
        instances[[i, 0]] = 0; // class ID
        instances[[i, 1]] = u64::MAX; // first y (inclusive)
        instances[[i, 2]] = u64::MIN; // last y (exclusive)
        instances[[i, 3]] = u64::MAX; // first x (inclusive)
        instances[[i, 4]] = u64::MIN; // last x (exclusive)
    }
    
    for y in 0..gt.shape[0] {
        let y64 = y as u64;
        
        for x in 0..gt.shape[1] {
            let i = inst_img[[y, x]];
            if i == 0 {
                continue;
            }
            
            let i = i - 1;
            let x64 = x as u64;
            
            instances[[i, 0]] = gt[[y, x]] as u64;
            instances[[i, 1]] = instances[[i, 1]].min(y64);
            instances[[i, 2]] = instances[[i, 2]].max(y64+1);
            instances[[i, 3]] = instances[[i, 3]].min(x64);
            instances[[i, 4]] = instances[[i, 4]].max(x64+1);
        }
    }
}
