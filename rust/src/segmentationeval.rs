use crate::utils::*;
use std::sync::Arc;

fn get_predicted_class(yp: &NdArray<f32>, i: i32, y: i32, x: i32) -> i32 {
    let mut result = 0;
    
    for c in 1..yp.shape[1] {
        if yp[[i, c, y, x]] > yp[[i, result, y, x]] {
            result = c;
        }
    }
    
    result
}

fn get_second_highest_probability(yp: &NdArray<f32>, i: i32, y: i32, x: i32, skip_index: i32) -> f32 {
    let mut result = if skip_index == 0 { 1 } else { 0 };
    
    for c in 1..yp.shape[1] {
        if c == skip_index {
            continue;
        }
        
        if yp[[i, c, y, x]] > yp[[i, result, y, x]] {
            result = c;
        }
    }
    
    yp[[i, result, y, x]]
}

fn is_boundary(yb: &NdArray<i32>, i: i32, cy: i32, cx: i32, size: i32, threshold: i32) -> u8 {
    let mut count = 0;
    
    for y in (cy-size).max(0)..(cy+size+1).min(yb.shape[1]) {
        for x in (cx-size).max(0)..(cx+size+1).min(yb.shape[2]) {
            if yb[[i, y, x]] != yb[[i, cy, cx]] {
                count += 1;
            }
        }
    }
    
    if count >= threshold { 1 } else { 0 }
}

fn compute_results_for_image(i: i32,
    p: &mut NdArrayMut<f32>, tp: &mut NdArrayMut<u8>, boundary: &mut NdArrayMut<u8>,
    yt: &NdArray<i32>, yp: &NdArray<f32>, yb: &NdArray<i32>,
    boundary_size: i32, boundary_threshold: i32) {
    
    for y in 0..yt.shape[1] {
        for x in 0..yt.shape[2] {
            let true_class = yt[[i, y, x]];
            let pred_class = get_predicted_class(yp, i, y, x);
            p[[i, y, x, 0]] = yp[[i, true_class, y, x]];
            let other_p = get_second_highest_probability(yp, i, y, x, pred_class);
            
            if true_class == pred_class {
                tp[[i, y, x]] = 1;
                p[[i, y, x, 0]] -= other_p;
                p[[i, y, x, 1]] = p[[i, y, x, 0]];
            } else {
                tp[[i, y, x]] = 0;
                p[[i, y, x, 0]] -= yp[[i, pred_class, y, x]];
                p[[i, y, x, 1]] = yp[[i, pred_class, y, x]];
                p[[i, y, x, 1]] -= other_p;
            }
            
            boundary[[i, y, x]] = is_boundary(yb, i, y, x, boundary_size, boundary_threshold);
        }
    }
}

//PYTHON_EXPORT
pub fn segeval_compute_results(
    p: NdArrayMut<f32>, tp: NdArrayMut<u8>, boundary: NdArrayMut<u8>,
    yt: NdArray<i32>, yp: NdArray<f32>, yb: NdArray<i32>,
    boundary_size: i32, boundary_threshold: i32, num_threads: i32) {
        
    assert!(boundary_size > 0);
    assert!(boundary_threshold > 0);
    assert!(num_threads > 0);
        
    ndarray!(from_ndarray: p); // required for satisfying lifetime check wrt to multi-threading
    ndarray!(from_ndarray: tp); // required for satisfying lifetime check wrt to multi-threading
    ndarray!(from_ndarray: boundary); // required for satisfying lifetime check wrt to multi-threading
    ndarray!(from_ndarray: yt); // required for satisfying lifetime check wrt to multi-threading
    ndarray!(from_ndarray: yp); // required for satisfying lifetime check wrt to multi-threading
    ndarray!(from_ndarray: yb); // required for satisfying lifetime check wrt to multi-threading
    
    let p = Arc::new(p);
    let tp = Arc::new(tp);
    let boundary = Arc::new(boundary);
    let yt = Arc::new(yt);
    let yp = Arc::new(yp);
    let yb = Arc::new(yb);        
        
    let mut threads = Vec::new();
        
    for thread_id in 0..num_threads {
        let p = p.clone();
        let tp = tp.clone();
        let boundary = boundary.clone();
        let yt = yt.clone();
        let yp = yp.clone();
        let yb = yb.clone();
        
        let handle = std::thread::spawn(move || {
            ndarray!(from_ndarray_mut: p);
            ndarray!(from_ndarray_mut: tp);
            ndarray!(from_ndarray_mut: boundary);
            
            for i in 0..p.shape[0] {
                if i % num_threads != thread_id {
                    continue;
                }
                
                compute_results_for_image(i, &mut p, &mut tp, &mut boundary, &yt, &yp, &yb, boundary_size, boundary_threshold);
            }
        });
        
        threads.push(handle);
    }

    while !threads.is_empty() {
        threads.pop().unwrap().join().unwrap();
    }
}

//PYTHON_EXPORT
pub fn segeval_integrate(p: NdArray<f32>) -> f64 {
    let n = p.shape[0];
    let mut result = 0.0;
    let denom = n as f64;
    
    for i in 0..n-1 {
        result += (((i+1) as f64) / denom) * ((p[i+1] as f64) - (p[i] as f64));
    }
    result += 1.0 - (p[n-1] as f64);
    
    result
}
