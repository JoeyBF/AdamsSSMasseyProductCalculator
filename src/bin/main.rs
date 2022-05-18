// cargo run --bin main

// import library root
use massey::*;

use fp::vector::FpVector;

use adams::AdamsMultiplication;

//pub mod computation;
//use computation::ComputationResult;

/* need to store the products
 * need to be able to extract Massey productable triples
 * then need to compute the Massey products and store them.
 * should be extensible
 * note Massey productable triples can involve non generators
 *
 * Multiplication is a bilinear map
 * Adams(s1,t1) x Adams(s2,t2) -> Adams(s1+s2,t1+t2)
 * Idea 1:
 * Store bilinear map as 3d matrix
 * (normal 2d matrix with entries in Adams(s1+s2,t1+t2))
 * Idea 2:
 * Store as linear map from the tensor product
 * Adams(s1,t1) \otimes Adams(s2,t2) -> Adams(s1+s2,t1+t2)
 * this is a normal matrix
 * Idea 3:
 * For each generator x_{s1,t1,i} in (s1, t1) store the matrix
 * for left multiplication x_{s1,t1,i}
 * (this is what we start with)
 * !!! We'll start with this !!! and adjust as necessary
 *
 * Goal is to compute pairs
 * (a,b) \in Adams(s1,t1)\times Adams(s2,t2) such that
 * mu(a,b) = 0
 *
 */

/*
impl Save for AdamsMultiplication {

}
*/

fn main() -> anyhow::Result<()> {
    let max_s = 20;
    let max_t = 40;
    //let mult_max_s=15;
    //let mult_max_t=30;
    //let mult_with_max_s=15;
    //let mult_with_max_t=30;

    println!("Loading and extending resolution...");
    let mut adams_mult: AdamsMultiplication = AdamsMultiplication::new()?;
    let prime = adams_mult.prime();

    adams_mult.extend_resolution_to((max_s, max_t).into())?;

    //fp::vector::initialize_limb_bit_index_table(adams_mult.resolution().prime());

    println!("Loading and computing multiplications...");
    match adams_mult.compute_all_multiplications() {
        Ok(_) => {}
        Err(err_info) => {
            eprintln!("{}", err_info);
        }
    }

    let h0 = (1, 1, FpVector::from_slice(prime, &[1])).into();
    let h1 = (1, 2, FpVector::from_slice(prime, &[1])).into();
    let max_massey_deg = (32, 102).into(); //(25,60).into();//(32, 96).into();
                                           // compute Massey products
                                           // <-,h0,h1>
                                           /*
                                           println!("Computing kernels for multiplication by h0 = {}...", h0);
                                           // first compute kernels for h0
                                           let kers_h0 = match adams_mult.compute_kernels_right_multiplication(&h0, max_massey_deg) {
                                               Ok(kers) => kers,
                                               Err(err_info) => {
                                                   eprintln!("{}", err_info);
                                                   // fail
                                                   return error::from_string(err_info);
                                                   //std::process::exit(-1);
                                               }
                                           };
                                           println!("Computing massey products <-,{},{}>...", h0, h1);
                                           let (deg_computed, massey_h0_h1) = adams_mult.compute_massey_prods_for_pair(&kers_h0, max_massey_deg, &h0, &h1);
                                           println!("Massey products <-,{},{}> computed through degree {} out of {}", h0, h1, deg_computed, max_massey_deg);
                                           let shift_deg = (1,3).into();
                                           for (a, rep, indet) in massey_h0_h1 {
                                               let rep_ae: AdamsElement = (a.degree() + shift_deg, rep).into();
                                               println!("<{}, h0, h1> = {} + {}", a, rep_ae, indet.matrix);
                                           }
                                           */
    println!("Computing kernels for multiplication by h1 = {}...", h1);
    // first compute kernels for h0
    let kers_h1 = match adams_mult.compute_kernels_right_multiplication(&h1, max_massey_deg) {
        Ok(kers) => kers,
        Err(err_info) => {
            eprintln!("{}", err_info);
            // fail
            return Err(anyhow::anyhow!(err_info));
            //std::process::exit(-1);
        }
    };
    println!("Computing massey products <-,{},{}>...", h1, h0);
    let (deg_computed, _) =
        adams_mult.compute_massey_prods_for_pair(&kers_h1, max_massey_deg, &h1, &h0);
    println!(
        "Massey products <-,{},{}> computed through degree {} out of {}",
        h1, h0, deg_computed, max_massey_deg
    );
    //let shift_deg = (1,3).into();

    // save stuff, disable for now, TODO
    /*
    let mut save_file = File::create(massey_product_save_file)?;
    deg_computed.save(&mut save_file)?;
    massey_h1_h0.len().save(&mut save_file)?;
    for (a, prod) in massey_h1_h0 {
        let rep_ae: AdamsElement = prod.rep_elt();
        println!("<{}, h1, h0> = {} + {}", a, rep_ae, prod.indet().matrix);
        a.save(&mut save_file)?;
        prod.save(&mut save_file)?;
    }
    */

    //adams_mult.compute_multiplications(mult_max_s, mult_max_t, mult_with_max_s, mult_with_max_t);
    //adams_mult.brute_force_compute_all_massey_products((7,30).into());

    /*
    println!("Iterate over whole F_2^5");
    for fp_vec in AllVectorsIterator::new_whole_space(adams_mult.prime(), 5) {
        println!("fp_vec: {}", fp_vec);
    }

    let p = adams_mult.prime();
    let input = [ vec![1, 0, 0, 1, 1]
                , vec![0, 1, 0, 1, 0]
                , vec![0, 0, 1, 0, 1]
                ];
    let mut m = Matrix::from_vec(p, &input);
    m.row_reduce();
    let subspace = Subspace {
        matrix: m
    };

    println!("Iterate over subspace");
    for fp_vec in AllVectorsIterator::new(&subspace) {
        println!("fp_vec: {}", fp_vec);
    }
    */

    //adams_mult.possible_nontrivial_massey_products();

    Ok(())
}
