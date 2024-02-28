use crate::utils::*;

//PYTHON_EXPORT
pub fn draw_circle(
    mut img: NdArray<u8>, mut gt: NdArray<i32>, center: NdArray<i32>, radius: f32,
    color: NdArray<u8>, class_index: i32, yrange: NdArray<i32>, xrange: NdArray<i32>
) {
    let radius = radius.powi(2);
    
    for y in yrange[0]..yrange[1] {
        let dy = ((center[0] - y) as i32).pow(2);
        
        for x in xrange[0]..xrange[1] {
            let dx = ((center[1] - x) as i32).pow(2);
            if ((dy + dx) as f32) > radius {
                continue;
            }
            
            for c in 0..3 {
                img[[y, x, c]] = color[c];
            }
            gt[[y, x]] = class_index;
        }
    }
}
