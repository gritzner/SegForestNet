use crate::utils::*;
use std::collections::{BTreeMap, BTreeSet};

//PYTHON_EXPORT
pub fn rgb2ids(mut gt_ids: NdArrayMut<u8>, gt_rgb: NdArray<u8>, mut inst_ids: NdArrayMut<i32>, inst_rgb: NdArray<u8>, class_map_array: NdArray<u8>) -> usize {
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
pub fn extract_instances(mut instances: NdArrayMut<u64>, mut rows: NdArrayMut<i32>, mut cols: NdArrayMut<i32>, gt: NdArray<u8>, inst_img: NdArray<i32>) {
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
    
    for i in 0..instances.shape[0] {
        let instance_id = (i + 1) as u64;
        
        for y in instances[[i, 1]]..instances[[i, 2]] {
            rows[[y, 0]] += 1;
            rows[[y, instance_id]] = 1;
        }
        
        for x in instances[[i, 3]]..instances[[i, 4]] {
            cols[[x, 0]] += 1;
            cols[[x, instance_id]] = 1;
        }
    }
    
    for y in 0..gt.shape[0] {
        let mut j = 1;
        for i in 1..rows.shape[1] {
            if rows[[y, i]] == 1 {
                rows[[y, j]] = i;
                j += 1;
            }
        }
        for i in j..rows.shape[1] {
            rows[[y, i]] = 0;
        }
    }
    
    for x in 0..gt.shape[1] {
        let mut j = 1;
        for i in 1..cols.shape[1] {
            if cols[[x, i]] == 1 {
                cols[[x, j]] = i;
                j += 1;
            }
        }
        for i in j..cols.shape[1] {
            cols[[x, i]] = 0;
        }
    }
}

pub fn convert_to_sparse_map(array: &NdArray<i32>) -> BTreeMap<i32, BTreeSet<i32>> {
    let mut map = BTreeMap::new();
    for i in 0..array.shape[0] {
        let mut instances = BTreeSet::new();
        for j in 0..array.shape[1] {
            let instance_id = array[[i, j]];
            if instance_id == 0 {
                break;
            }
            instances.insert(instance_id);
        }
        map.insert(i, instances);
    }
    
    map
}

pub fn get_pixel_sparse(y: i32, x: i32, instances: &NdArray<u64>, row: &BTreeSet<i32>, cols: &NdArray<i32>, masks: &NdArray<u8>, return_class: bool) -> i32 {
    for i in 0..cols.shape[1] {
        let instance_id = cols[[x, i]];
        if instance_id == 0 {
            break;
        }
        if !row.contains(&instance_id) {
            continue;
        }
        
        let instance_index = instance_id - 1;        
        let mask_y = y - (instances[[instance_index, 1]] as i32);
        let mask_x = x - (instances[[instance_index, 3]] as i32);
        let mask_width = (instances[[instance_index, 4]] - instances[[instance_index, 3]]) as i32;
        let index = (instances[[instance_index, 5]] as i32) + mask_y * mask_width + mask_x;
        
        if masks[index] != 0 {
            return if return_class { instances[[instance_index, 0]] as i32 } else { instance_id };
        }
    }
    
    0
}
