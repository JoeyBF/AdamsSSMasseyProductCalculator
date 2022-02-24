use anyhow;
use dashmap::DashMap;
use rayon::prelude::*;

use std::cmp::Ordering;
use std::fs::DirBuilder;

use std::clone::Clone;
use std::path::PathBuf;
use std::sync::Arc;

use algebra::module::homomorphism::ModuleHomomorphism;
use algebra::module::Module;

use ext::chain_complex::{ChainComplex, ChainHomotopy, FreeChainComplex};

#[cfg(feature = "nassau")]
use ext::nassau::Resolution as NassauResolution;
#[cfg(not(feature = "nassau"))]
use ext::resolution::Resolution as ExtResolution;

use ext::resolution_homomorphism::ResolutionHomomorphism;

#[cfg(not(feature = "nassau"))]
use ext::CCC;

use fp::matrix::Matrix;
use fp::matrix::Subspace;
use fp::prime::ValidPrime;
use fp::vector::FpVector;

use std::io;

use super::{AdamsElement, AdamsGenerator, Bidegree, MasseyProduct};

use crate::lattice::{meet, JoinSemilattice, MeetSemilattice};
use crate::utils::{get_max_defined_degree, AllVectorsIterator};

#[cfg(feature = "nassau")]
type Resolution = NassauResolution;
#[cfg(not(feature = "nassau"))]
type Resolution = ExtResolution<CCC>;

//#[derive(Clone)]
pub struct AdamsMultiplication {
    /// the resolution object
    resolution: Arc<Resolution>,
    /// filename storing resolution data
    res_file_name: String,
    ///directory to incrementally store resolutions to save progress
    res_data_directory: Option<String>,
    /// max_s w/ dimensions computed
    max_s: u32,
    /// max_t w/ dimensions computed
    max_t: i32,
    /// keeps track of which degrees the multiplication by
    /// a given basis element (s,t,index) is computed for
    //multiplication_range_computed:
    //    DashMap<AdamsGenerator, Bidegree>,
    /// stores the multiplication matrices for each degree
    /// where we could compute the multiplication
    multiplication_matrices: DashMap<AdamsGenerator, (Bidegree, DashMap<Bidegree, Matrix>)>,
    /// directory to store computed multiplications
    /// each file will have the multiplication matrices as computed so far
    /// it's not great. We'll have to see how we can make this more efficient
    /// possibly need to go in to ext and add Save/Load to resolution homomorphisms
    multiplication_data_directory: Option<String>,
    resolution_homomorphism_data_directory: Option<String>,
    /// directory to store massey product outputs
    massey_product_data_directory: Option<String>,
}

impl AdamsMultiplication {
    pub fn max_deg(&self) -> Bidegree {
        (self.max_s, self.max_t).into()
    }

    /// largest s for which (n,s) might be nonzero
    fn max_nonzero_s(&self, n: u32) -> Option<u32> {
        if n == 0 {
            None
        } else {
            // max nonzero s is between
            // (n-3)/2 and (n-1)/2
            // which is (n-1)/2 and (n-1)/2-1
            let max_s_0 = (n - 1) / 2; // any index greater than this will definitely be 0
            let n_res = n % 8;
            let max_s = if n_res == 0 || n_res == 5 || n_res == 7 {
                max_s_0 - 1 // in these cases, we know that (n, max_s_0) will definitely be 0, so decrease max_s_0 by 1
            } else {
                max_s_0
            };
            Some(max_s)
        }
    }

    pub fn is_stem_fully_computed(&self, n: u32) -> bool {
        match self.max_nonzero_s(n) {
            Some(s) => s <= self.max_s,
            None => false,
        }
    }

    pub fn max_fully_computed_stem(&self) -> Option<u32> {
        // stem n is fully computed if we have computed
        // (n,0)..(n,s) for all s where n >= 2s+epsilon
        // where epsilon is 1 for s 0,1 mod 4, 2 for s 2 mod 4, and 3 for s 3 mod 4
        // i.e. s where s <= (n-epsilon)/2
        // or for degrees (t,s) with t=s+n for all nonnegative s with t-s >= 2s+epsilon
        // or s with s < (t-epsilon)/3
        // so we get (t,s) degrees
        // (n,0), (n-1,1), (n-2, 2) ... (n-max_s(n), max_s(n))
        let mut fully_computed = None;
        for n in 1..=self.max_t as u32 {
            if !self.is_stem_fully_computed(n) {
                break;
            } else {
                fully_computed = Some(n);
            }
        }
        fully_computed
    }

    fn ensure_resolution_data_directory_exists(&self) -> io::Result<bool> {
        if !self.has_resolution_data_directory() {
            return Ok(false);
        }
        DirBuilder::new().recursive(true).create(
            self.res_data_directory
                .clone()
                .expect("resolution data directory unexpectedly None"),
        )?;
        Ok(true)
    }

    fn ensure_multiplication_data_directory_exists(&self) -> io::Result<bool> {
        if !self.has_multiplication_data_directory() {
            return Ok(false);
        }
        DirBuilder::new().recursive(true).create(
            self.multiplication_data_directory
                .clone()
                .expect("multiplication data directory unexpectedly None"),
        )?;
        Ok(true)
    }

    fn ensure_resolution_homomorphism_data_directory_exists(&self) -> io::Result<bool> {
        if !self.has_resolution_homomorphism_data_directory() {
            return Ok(false);
        }
        DirBuilder::new().recursive(true).create(
            self.multiplication_data_directory
                .clone()
                .expect("multiplication data directory unexpectedly None"),
        )?;
        Ok(true)
    }

    pub fn extend_resolution_to(&mut self, deg: Bidegree) -> io::Result<()> {
        if deg <= self.max_deg() {
            return Ok(());
        }

        let mut cur_deg = self.max_deg();

        while cur_deg < deg {
            if cur_deg.s() < deg.s() {
                *cur_deg.s_mut() += 1;
            } else {
                *cur_deg.t_mut() += 1;
            }

            let (s, t) = cur_deg.into();
            eprintln!("Extending to degree {}", cur_deg);

            self.resolution.compute_through_bidegree(s, t);

            /*
            if self.has_resolution_data_directory() {
                let file: File = File::create(self.resolution_file_path(cur_deg).expect("unexpectedly resolution file path was None"))?;
                let mut buf_file = std::io::BufWriter::new(file);
                self.resolution.save(&mut buf_file)?;
            }
            */
        }
        /*
        let file: File = File::create(&self.res_file_name)?;
        let mut buf_file = std::io::BufWriter::new(file);
        self.resolution.save(&mut buf_file)?;
        */
        // update max degree
        self.max_s = deg.s;
        self.max_t = deg.t;
        Ok(())
    }

    pub fn new(
        res_file_name: String,
        res_data_directory: Option<String>,
        multiplication_data_directory: Option<String>,
        resolution_homomorphism_data_directory: Option<String>,
        massey_product_data_directory: Option<String>,
    ) -> anyhow::Result<AdamsMultiplication> {
        let save_path = PathBuf::from(&res_file_name);

        let (max_s, max_t) = get_max_defined_degree(&save_path);
        eprintln!("Max degree detected: ({}, {})", max_s, max_t);

        #[cfg(not(feature = "nassau"))]
        let res = Arc::new({
            let mut res = construct("S_2", Some(save_path))?;
            res.load_quasi_inverse = false;
            res
        });

        #[cfg(feature = "nassau")]
        let res = Arc::new(Resolution::new(Some(save_path)));

        #[cfg(feature = "stem")]
        res.compute_through_stem(max_s, max_t - max_s as i32);

        #[cfg(not(feature = "stem"))]
        res.compute_through_bidegree(max_s, max_t);

        let result = AdamsMultiplication {
            resolution: res,
            res_file_name,
            res_data_directory,
            max_s,
            max_t,
            multiplication_matrices: DashMap::new(),
            multiplication_data_directory,
            resolution_homomorphism_data_directory,
            massey_product_data_directory,
        };

        result.ensure_multiplication_data_directory_exists()?;
        result.ensure_resolution_data_directory_exists()?;
        result.ensure_resolution_homomorphism_data_directory_exists()?;

        Ok(result)
    }

    /// return Arc to the resolution
    pub fn resolution(&self) -> Arc<Resolution> {
        self.resolution.clone()
    }

    pub fn prime(&self) -> ValidPrime {
        self.resolution.prime()
    }

    /// return copy of the filename
    pub fn res_file_name(&self) -> String {
        self.res_file_name.clone()
    }

    /// return nonmutable reference
    pub fn multiplication_matrices(
        &self,
    ) -> &DashMap<AdamsGenerator, (Bidegree, DashMap<Bidegree, Matrix>)> {
        &self.multiplication_matrices
    }

    pub fn num_gens(&self, s: u32, t: i32) -> Option<usize> {
        if self.resolution.has_computed_bidegree(s, t) {
            Some(self.resolution.number_of_gens_in_bidegree(s, t))
        } else {
            None
        }
        //self.num_gens.get(&(s,t).into()).map(|u| -> usize { u.clone() })
    }
    pub fn num_gens_bidegree(&self, deg: Bidegree) -> Option<usize> {
        let (s, t) = deg.into();
        self.num_gens(s, t)
    }
    pub fn multiplication_in_bounds(&self, deg1: Bidegree, deg2: Bidegree) -> bool {
        let (s1, t1) = deg1.into();
        let (s2, t2) = deg2.into();
        if (t1 < 0) || (t2 < 0) {
            false
        } else {
            (s1 + s2 < self.max_s) && (t1 + t2 < self.max_t)
        }
    }

    pub fn adams_gen_to_resoln_hom(
        &self,
        g: AdamsGenerator,
    ) -> Result<ResolutionHomomorphism<Resolution, Resolution>, String> {
        let (s, t, idx) = g.into();
        let hom = ResolutionHomomorphism::new(
            format!("({},{},{})", s, t, idx),
            self.resolution(),
            self.resolution(),
            s,
            t,
        );
        let dim = self
            .num_gens(s, t)
            .ok_or(format!("resolution not computed through ({}, {})", s, t))?;
        let mut matrix = Matrix::new(self.prime(), dim, 1);
        matrix[idx].set_entry(0, 1);
        hom.extend_step(s, t, Some(&matrix));
        Ok(hom)
    }

    #[cfg(feature = "save-res")]
    pub fn try_load_resoln_hom_for_adams_gen(
        &self,
        g: AdamsGenerator,
    ) -> io::Result<Option<ResolutionHomomorphism<Resolution, Resolution>>> {
        let path = match self.multiplication_hom_file_path(g) {
            Some(path) => path,
            None => return Ok(None), // no directory provided
        };

        if path.exists() {
            let mut file = File::open(path)?;
            let res_hom = <ResolutionHomomorphism<Resolution, Resolution> as Load>::load(
                &mut file,
                &(self.resolution.clone(), self.resolution.clone()),
            )?;
            Ok(Some(res_hom))
        } else {
            // this is actually fine though
            Ok(None)
        }
    }

    #[cfg(not(feature = "save-res"))]
    pub fn try_load_resoln_hom_for_adams_gen(
        &self,
        _g: AdamsGenerator,
    ) -> io::Result<Option<ResolutionHomomorphism<Resolution, Resolution>>> {
        Ok(None)
    }

    #[cfg(feature = "save-res")]
    pub fn save_resoln_hom_for_adams_gen(
        &self,
        g: AdamsGenerator,
        res_hom: &ResolutionHomomorphism<Resolution, Resolution>,
    ) -> io::Result<()> {
        let path = self.multiplication_hom_file_path(g).ok_or(io::Error::new(
            io::ErrorKind::Other,
            "No resolution homomorphism data directory, can't save resolution homomorphisms.",
        ))?;
        let mut file = File::create(path)?;
        res_hom.save(&mut file)
    }

    #[cfg(not(feature = "save-res"))]
    pub fn save_resoln_hom_for_adams_gen(
        &self,
        _g: AdamsGenerator,
        _res_hom: &ResolutionHomomorphism<Resolution, Resolution>,
    ) -> io::Result<()> {
        Ok(())
    }

    pub fn adams_elt_to_resoln_hom(
        &self,
        e: &AdamsElement,
    ) -> ResolutionHomomorphism<Resolution, Resolution> {
        let (s, t, v) = e.into();
        let hom = ResolutionHomomorphism::new(
            format!("({},{},{})", s, t, v),
            self.resolution(),
            self.resolution(),
            s,
            t,
        );
        let mut matrix = Matrix::new(v.prime(), v.len(), 1);
        for idx in 0..v.len() {
            matrix[idx].set_entry(0, v.entry(idx));
        }
        hom.extend_step(s, t, Some(&matrix));
        hom
    }

    pub fn has_multiplication_data_directory(&self) -> bool {
        self.multiplication_data_directory.is_some()
    }
    pub fn has_resolution_data_directory(&self) -> bool {
        self.res_data_directory.is_some()
    }
    pub fn has_massey_product_data_directory(&self) -> bool {
        self.massey_product_data_directory.is_some()
    }
    pub fn has_resolution_homomorphism_data_directory(&self) -> bool {
        self.resolution_homomorphism_data_directory.is_some()
    }

    #[allow(dead_code)]
    fn multiplication_file_name(g: AdamsGenerator) -> String {
        format!("mult_s{}_t{}_{}.data", g.s(), g.t(), g.idx())
    }

    #[allow(dead_code)]
    fn multiplication_file_path(&self, g: AdamsGenerator) -> Option<PathBuf> {
        match &self.multiplication_data_directory {
            Some(dir) => {
                let path: PathBuf = [dir.clone(), Self::multiplication_file_name(g)]
                    .iter()
                    .collect();
                Some(path)
            }
            None => None,
        }
    }

    #[allow(dead_code)]
    fn multiplication_hom_file_name(g: AdamsGenerator) -> String {
        format!("mult_s{}_t{}_{}_homomorphism.data", g.s(), g.t(), g.idx())
    }

    #[allow(dead_code)]
    fn multiplication_hom_file_path(&self, g: AdamsGenerator) -> Option<PathBuf> {
        match &self.resolution_homomorphism_data_directory {
            Some(dir) => {
                let path: PathBuf = [dir.clone(), Self::multiplication_hom_file_name(g)]
                    .iter()
                    .collect();
                Some(path)
            }
            None => None,
        }
    }

    #[allow(dead_code)]
    fn resolution_file_name(deg: Bidegree) -> String {
        format!("S_2_resolution_s{}_t{}.data", deg.s(), deg.t())
    }

    #[allow(dead_code)]
    fn resolution_file_path(&self, deg: Bidegree) -> Option<PathBuf> {
        match &self.res_data_directory {
            Some(dir) => {
                let path: PathBuf = [dir.clone(), Self::resolution_file_name(deg)]
                    .iter()
                    .collect();
                Some(path)
            }
            None => None,
        }
    }

    //TODO had to disable loading temporarily
    pub fn load_multiplications_for(
        &self,
        _g: AdamsGenerator,
    ) -> io::Result<Option<(Bidegree, DashMap<Bidegree, Matrix>)>> {
        Ok(None)
        /*
        let path = match self.multiplication_file_path(g) {
            Some(path) => path,
            None => return Ok(None) // no directory provided
        };
        if path.exists() {
            let mut file = File::open(path)?;
            let max_deg = Bidegree::load(&mut file, &())?;
            let hm: DashMap<Bidegree, Matrix> = LoadHM::load(&mut file, &((), self.prime()))?.into();
            Ok(Some((max_deg, hm)))
        } else {
            // this is actually fine though
            Ok(None)
        }
        */
    }

    //TODO disabled saving
    pub fn save_multiplications_for(
        &self,
        _g: AdamsGenerator,
        _computed_range: Bidegree,
        _matrices: &DashMap<Bidegree, Matrix>,
    ) -> io::Result<()> {
        Ok(())
        /*
        let path = self.multiplication_file_path(g).ok_or_else(|| io::Error::new(io::ErrorKind::Other, "No multiplication data directory, can't save multiplications."))?;
        let mut file = File::create(path)?;
        computed_range.save(&mut file)?;
        SaveHM(matrices).save(&mut file)?;
        Ok(())
        */
    }

    #[cfg(feature = "stem")]
    pub fn compute_multiplication(
        &self,
        g: AdamsGenerator,
        mult_with_max: Bidegree,
    ) -> Result<(Bidegree, DashMap<Bidegree, Matrix>), String> {
        eprintln!("Computing multiplication for {}", g);
        let (s, t, idx) = g.into();
        let n = t - s as i32;
        let g_deg = g.degree();
        let max_deg = self.max_deg();
        match g_deg.partial_cmp(&max_deg) {
            None | Some(Ordering::Greater) => {
                return Err(format!(
                    "({}, {}, {}) is out of computed range: ({}, {})",
                    s, t, idx, self.max_s, self.max_t
                ))
            }
            _ => {}
        };

        let rmax_poss_s = self.max_s - s as u32;
        let rmax_poss_t = self.max_t - t as i32;
        let rmax_poss = (rmax_poss_s, rmax_poss_t).into();

        let actual_rmax = mult_with_max.meet(rmax_poss);

        let hm = DashMap::new();

        let compute_to = actual_rmax;

        let hom = self.adams_gen_to_resoln_hom(g)?;

        let compute_to_s = compute_to.s();
        let compute_to_n = compute_to.n();

        // extend hom

        hom.extend_through_stem(compute_to_s + s, compute_to_n + n);

        // hom has been extended
        // save the hom first
        if self.has_resolution_homomorphism_data_directory() {
            match self.save_resoln_hom_for_adams_gen(g, &hom) {
                Ok(()) => (),
                Err(err) => {
                    eprintln!(
                        "Failed to save resolution homomorphism for generator {}.\nError: {}",
                        g, err
                    );
                }
            };
        }

        // then read off and insert the multiplication matrices

        // ok let's do the proper multiplications
        for rhs in compute_to.iter_stem() {
            // might want to iterate over stems, since these are the only nonzero ones, but for now
            // we just skip those with n negative
            if rhs.n() < 0 {
                continue; // rhs trivially empty, since n is negative
            }
            let target_deg = (s + rhs.s(), t + rhs.t()).into();
            let dim_rhs = match self.num_gens_bidegree(rhs) {
                Some(n) => n,
                None => {
                    return Err(format!(
                        "Dimension at rhs {} not computed. Expected for computing multiplication by {} in degree {}",
                        rhs,
                        g,
                        rhs,
                    ));
                }
            };
            let dim_target = match self.num_gens_bidegree(target_deg) {
                Some(n) => n,
                None => {
                    return Err(format!(
                        "Dimension at target {} not computed. Expected for computing multiplication by {} in degree {}",
                        target_deg,
                        g,
                        rhs,
                    ));
                } // this is an error
            };
            if dim_rhs == 0 || dim_target == 0 {
                continue; // nothing in domain, or nothing in codomain, multiplication is trivially 0
                          // store nothing
            }
            eprintln!("with {}", rhs);
            //let gens2 = &module2.gen_names()[j2];
            let matrix = hom.get_map(target_deg.s()).hom_k(rhs.t());
            // convert to fp::matrix::Matrix and store
            hm.insert(rhs, Matrix::from_vec(self.prime(), &matrix));
        }
        if self.has_multiplication_data_directory() {
            match self.save_multiplications_for(g, compute_to, &hm) {
                Ok(_) => {}
                Err(e) => {
                    return Err(e.to_string());
                }
            };
        }

        Ok((compute_to, hm))
    }

    #[cfg(not(feature = "stem"))]
    pub fn compute_multiplication(
        &self,
        g: AdamsGenerator,
        mult_with_max: Bidegree,
    ) -> Result<(Bidegree, DashMap<Bidegree, Matrix>), String> {
        eprintln!("Computing multiplication for {}", g);
        let (s, t, idx) = g.into();
        let g_deg = g.degree();
        let max_deg = self.max_deg();
        match g_deg.partial_cmp(&max_deg) {
            None | Some(Ordering::Greater) => {
                return Err(format!(
                    "({}, {}, {}) is out of computed range: ({}, {})",
                    s, t, idx, self.max_s, self.max_t
                ))
            }
            _ => {}
        };
        //let (rmax_s, rmax_t) = mult_with_max.into();
        let rmax_poss_s = self.max_s - s as u32;
        let rmax_poss_t = self.max_t - t as i32;
        let rmax_poss = (rmax_poss_s, rmax_poss_t).into();

        //let actual_rmax_s = min(rmax_s, rmax_poss_s);
        //let actual_rmax_t = min(rmax_t, rmax_poss_t);
        let actual_rmax = mult_with_max.meet(rmax_poss);

        let already_computed = match self.load_multiplications_for(g) {
            Ok(stuff) => stuff,
            Err(e) => {
                return Err(e.to_string());
            }
        };

        let (partially_computed, computed_range, hm) = match already_computed {
            Some((range, hm)) => (true, range, hm),
            None => (false, (0, 0).into(), DashMap::new()),
        };

        let compute_to = if partially_computed {
            // determine what's left to compute
            if actual_rmax <= computed_range {
                // done, can't compute more
                return Ok((computed_range, hm));
            }
            // this should be true, or we're going to run into problems
            // might remove this restriction later
            assert!(rmax_poss >= computed_range);
            computed_range.join(actual_rmax) // should compute a strictly larger rectangle
        } else {
            actual_rmax
        };
        let hom = match self.try_load_resoln_hom_for_adams_gen(g) {
            Ok(Some(hom)) => hom,
            Err(_err) => {
                // ignore io error and recompute
                // TODO
                self.adams_gen_to_resoln_hom(g)?
            }
            // file not found, definitely recompute
            Ok(None) => self.adams_gen_to_resoln_hom(g)?,
        };

        let (compute_to_s, compute_to_t) = compute_to.into();

        // extend hom

        hom.extend(compute_to_s + s, compute_to_t + t);

        // hom has been extended
        // save the hom first
        if self.has_resolution_homomorphism_data_directory() {
            match self.save_resoln_hom_for_adams_gen(g, &hom) {
                Ok(()) => (),
                Err(err) => {
                    eprintln!(
                        "Failed to save resolution homomorphism for generator {}.\nError: {}",
                        g, err
                    );
                }
            };
        }

        // then read off and insert the multiplication matrices

        // ok let's do the proper multiplications
        for rhs in compute_to.iter_s_t() {
            // might want to iterate over stems, since these are the only nonzero ones, but for now
            // we just skip those with n negative
            if rhs.n() < 0 {
                continue; // rhs trivially empty, since n is negative
            }
            if partially_computed && rhs <= computed_range {
                continue; // already computed, no need to compute again
            }
            let target_deg = (s + rhs.s(), t + rhs.t()).into();
            let dim_rhs = match self.num_gens_bidegree(rhs) {
                Some(n) => n,
                None => {
                    return Err(format!(
                        "Dimension at rhs {} not computed. Expected for computing multiplication by {} in degree {}",
                        rhs,
                        g,
                        rhs,
                    ));
                }
            };
            let dim_target = match self.num_gens_bidegree(target_deg) {
                Some(n) => n,
                None => {
                    return Err(format!(
                        "Dimension at target {} not computed. Expected for computing multiplication by {} in degree {}",
                        target_deg,
                        g,
                        rhs,
                    ));
                } // this is an error
            };
            if dim_rhs == 0 || dim_target == 0 {
                continue; // nothing in domain, or nothing in codomain, multiplication is trivially 0
                          // store nothing
            }
            eprintln!("with {}", rhs);
            //let gens2 = &module2.gen_names()[j2];
            let matrix = hom.get_map(target_deg.s()).hom_k(rhs.t());
            // convert to fp::matrix::Matrix and store
            hm.insert(rhs, Matrix::from_vec(self.prime(), &matrix));
        }
        if self.has_multiplication_data_directory() {
            match self.save_multiplications_for(g, compute_to, &hm) {
                Ok(_) => {}
                Err(e) => {
                    return Err(e.to_string());
                }
            };
        }

        Ok((compute_to, hm))
    }

    pub fn compute_all_multiplications(&mut self) -> Result<(), String> {
        //self.compute_multiplications(self.max_s, self.max_t, self.max_s, self.max_t);
        self.compute_multiplications(self.max_deg(), self.max_deg())
    }

    pub fn compute_multiplications(
        &mut self,
        lhs_max: Bidegree,
        rhs_max: Bidegree,
    ) -> Result<(), String> {
        let lhs_max = lhs_max.meet(self.max_deg()); // don't go out of range
        let rhs_max = rhs_max.meet(self.max_deg()); // don't go out of range
        for lhs in lhs_max.iter_s_t() {
            let dim = match self.num_gens_bidegree(lhs) {
                Some(n) => n,
                None => {
                    return Err(format!(
                        "compute_multiplications: Expected {} to be computed!",
                        lhs
                    ))
                }
            };
            for idx in 0..dim {
                let g = (lhs, idx).into();
                let mult_data = self.compute_multiplication(g, rhs_max)?;
                self.multiplication_matrices.insert(g, mult_data);
            }
        }
        Ok(())
    }

    /// boolean variable store determines whether or not to store the multiplication matrices
    /// callback called with adams generator, max bidegree for which multiplication matrices were computed
    /// and the hashmap of multiplication matrices for multiplication by the adams generator
    pub fn compute_all_multiplications_callback<F>(
        &mut self,
        store: bool,
        callback: F,
    ) -> Result<(), String>
    where
        F: Fn(AdamsGenerator, Bidegree, &DashMap<Bidegree, Matrix>) -> Result<(), String>
            + Send
            + Sync,
    {
        self.compute_multiplications_callback(self.max_deg(), self.max_deg(), store, callback)
    }

    /// boolean variable store determines whether or not to store the multiplication matrices
    /// callback called with adams generator, max bidegree for which multiplication matrices were computed
    /// and the hashmap of multiplication matrices for multiplication by the adams generator
    pub fn compute_multiplications_callback<F>(
        &mut self,
        lhs_max: Bidegree,
        rhs_max: Bidegree,
        store: bool,
        callback: F,
    ) -> Result<(), String>
    where
        F: Fn(AdamsGenerator, Bidegree, &DashMap<Bidegree, Matrix>) -> Result<(), String>
            + Send
            + Sync,
    {
        let s_line = query::optional("Compute s-line?", str::parse::<u32>);
        let lhs_max = lhs_max.meet(self.max_deg()); // don't go out of range
        let rhs_max = rhs_max.meet(self.max_deg()); // don't go out of range
        let f = Arc::new(callback);
        lhs_max
            .iter_s_t()
            .filter(|bideg| match s_line {
                Some(s) => bideg.s() == s,
                None => true,
            })
            .par_bridge()
            .map(|lhs| {
                let dim = match self.num_gens_bidegree(lhs) {
                    Some(n) => n,
                    None => {
                        return Err(format!(
                            "compute_multiplications: Expected {} to be computed!",
                            lhs
                        ))
                    }
                };
                (0..dim)
                    .into_par_iter()
                    .map(|idx| {
                        let g = (lhs, idx).into();
                        let mult_data = self.compute_multiplication(g, rhs_max)?;
                        f(g, mult_data.0, &mult_data.1)?;
                        if store {
                            self.multiplication_matrices.insert(g, mult_data);
                        }
                        Result::<(), String>::Ok(())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(())
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|_| ())
    }

    /// only uses the dimensions, none of the multiplicative structure
    pub fn possible_nontrivial_massey_products(&self) {
        let mut count = 0;
        let mut not_all_dim_1 = 0;
        let mut max_triples = 0;
        for s1 in 1..self.max_s {
            for t1 in s1 as i32..self.max_t {
                let n1 = match self.num_gens(s1, t1) {
                    Some(n) => n,
                    None => {
                        continue;
                    }
                };
                if n1 == 0 {
                    continue; // empty bidegree, massey products will be uninteresting
                }
                for s2 in 1..self.max_s {
                    for t2 in s2 as i32..self.max_t {
                        let n2 = match self.num_gens(s2, t2) {
                            Some(n) => n,
                            None => {
                                continue;
                            }
                        };
                        if n2 == 0 {
                            continue; // empty bidegree, massey products will be uninteresting
                        }

                        let shift_s = s1 + s2 - 1;
                        let shift_t = t1 + t2;

                        if shift_s > self.max_s {
                            continue;
                        }

                        for s3 in 1..self.max_s - shift_s {
                            for t3 in s3 as i32..self.max_t - shift_t {
                                let n3 = match self.num_gens(s3, t3) {
                                    Some(n) => n,
                                    None => {
                                        continue;
                                    }
                                };
                                if n3 == 0 {
                                    continue; // empty bidegree, massey products will be uninteresting
                                }
                                let final_s = shift_s + s3;
                                let final_t = shift_t + t3;
                                let target_n = match self.num_gens(final_s, final_t) {
                                    Some(n) => n,
                                    None => {
                                        continue;
                                    }
                                };
                                if target_n == 0 {
                                    continue;
                                }
                                if n1 * n2 * n3 > 1 {
                                    println!("Potential massey products ({},{}) x ({}, {}) x ({}, {}) -> ({}, {})", s1, t1, s2, t2, s3, t3, final_s, final_t);
                                    println!(
                                        "Dimensions: {} x {} x {} -> {}",
                                        n1, n2, n3, target_n
                                    );
                                    not_all_dim_1 += 1;
                                }
                                count += 1;
                                max_triples += n1 * n2 * n3;
                            }
                        }
                    }
                }
            }
        }
        println!("VS triples: {}", count);
        println!("VS triples not all linear: {}", not_all_dim_1);
        println!("Max elt triples: {}", max_triples);
    }

    pub fn left_multiplication_by(
        &self,
        l: Bidegree,
        vec: &FpVector,
        r: Bidegree,
    ) -> Result<Matrix, String> {
        let p = self.prime();
        let (ls, lt) = l.into();
        let (rs, rt) = r.into();
        let (gs, gt) = (ls + rs, lt + rt);
        self.num_gens_bidegree(l)
            .ok_or(format!("Couldn't get gens in left bidegree ({}, {})", ls , lt))
            .and_then(|dim_l| -> Result<Matrix, String> {
            assert_eq!(dim_l, vec.len()); // vec has to have same length as number of gens in bidegree l

            self.num_gens_bidegree(r)
                .ok_or(format!("Couldn't get gens in right bidegree ({}, {})", rs , rt))
                .and_then(|dim_r| -> Result<Matrix, String> {
                self.num_gens(gs, gt)
                    .ok_or(format!("Couldn't get gens in final bidegree ({}, {})", gs, gt))
                    .and_then(|dim_g| -> Result<Matrix, String> {
                    let mut result = Matrix::new(p, dim_r, dim_g);
                    if (dim_l==0) || (dim_r==0) || (dim_g==0) {
                        Ok(result)
                    } else {
                        for ix in 0..dim_l {
                            let coeff = vec.entry(ix);
                            if coeff == 0 {
                                continue;
                            }
                            let matrix = match self.multiplication_matrices.get(&(ls, lt, ix).into()) {
                                Some(kv) => {
                                    let hm = &kv.1;
                                    match hm.get(&r) {
                                        Some(m) => m.value().clone(),
                                        None => {
                                            return Err(format!("Couldn't get multiplication matrix for gen ({}, {}, {}) in bidegree ({}, {})", ls, lt, ix, rs, rt));
                                        }
                                    }
                                },
                                None => {
                                    return Err(format!("Couldn't get multiplication matrices for gen ({}, {}, {})", ls, lt, ix)); // couldn't find an important multiplication matrix
                                }
                            };
                            result += /* coeff* */ &matrix; // coeff is 1 though, so we're good
                        }
                        Ok(result)
                    }
                })
            })
        })
    }

    pub fn right_multiplication_by(
        &self,
        r: Bidegree,
        vec: &FpVector,
        l: Bidegree,
    ) -> Result<Matrix, String> {
        // TODO right now I'm assuming that the multiplication in the Adams Spectral Sequence is commutative
        // so we can return
        self.left_multiplication_by(r, vec, l)
        // Check this assumption
    }

    /// Indeterminacy of massey product only depends on the bidegree of the middle term.
    /// returns the pair of subspaces of  
    /// aR, Rc in R
    /// in the bidegree where <a,b,c> lives
    pub fn compute_indeterminacy_of_massey_product(
        &self,
        a: &AdamsElement,
        b: Bidegree,
        c: &AdamsElement,
    ) -> Result<(Subspace, Subspace), String> {
        let (s1, t1, v1) = a.into();
        let (s2, t2) = b.into();
        let (s3, t3, v3) = c.into();
        let (s_left, t_left) = (s1 + s2 - 1, t1 + t2);
        let (s_right, t_right) = (s2 + s3 - 1, t2 + t3);
        let (s_tot, t_tot) = (s1 + s2 + s3 - 1, t1 + t2 + t3);
        let p = self.prime();

        let left_dim = match self.num_gens(s_left, t_left) {
            Some(n) => n,
            None => {
                return Err(format!(
                    "Couldn't get dimension of ({}, {})",
                    s_left, t_left
                ));
            }
        };
        let right_dim = match self.num_gens(s_right, t_right) {
            Some(n) => n,
            None => {
                return Err(format!(
                    "Couldn't get dimension of ({}, {})",
                    s_right, t_right
                ));
            }
        };
        let total_dim = match self.num_gens(s_tot, t_tot) {
            Some(n) => n,
            None => {
                return Err(format!("Couldn't get dimension of ({}, {})", s_tot, t_tot));
            }
        };

        let l_indet = if right_dim == 0 {
            Subspace::empty_space(p, total_dim)
        } else {
            // compute left multiplication
            // from (s_right, t_right) -> (s_tot, t_tot)
            let left_indet = match self.left_multiplication_by(
                (s1, t1).into(),
                v1,
                (s_right, t_right).into(),
            ) {
                Ok(lmat) => lmat,
                Err(err) => {
                    return Err(
                        format!(
                            "Couldn't compute the left multiplication ({}, {}, {})* : ({}, {}) -> ({}, {}) because {}", 
                            s1, t1, v1,
                            s_right, t_right,
                            s_tot, t_tot,
                            err
                        ));
                }
            };

            let (l_aug_start, mut l_indet_aug) =
                Matrix::augmented_from_vec(p, &left_indet.to_vec());
            l_indet_aug.row_reduce();

            l_indet_aug.compute_image(left_indet.columns(), l_aug_start)
        };

        let r_indet = if left_dim == 0 {
            Subspace::empty_space(p, total_dim)
        } else {
            // compute left multiplication
            // from (s_right, t_right) -> (s_tot, t_tot)
            let right_indet = match self.right_multiplication_by(
                (s3, t3).into(),
                v3,
                (s_left, t_left).into(),
            ) {
                Ok(rmat) => rmat,
                Err(err) => {
                    return Err(
                        format!(
                            "Couldn't compute the right multiplication *({}, {}, {}) : ({}, {}) -> ({}, {}) because {}", 
                            s3, t3, v3,
                            s_left, t_left,
                            s_tot, t_tot,
                            err
                        ));
                }
            };

            let (r_aug_start, mut r_indet_aug) =
                Matrix::augmented_from_vec(p, &right_indet.to_vec());
            r_indet_aug.row_reduce();

            r_indet_aug.compute_image(right_indet.columns(), r_aug_start)
        };

        Ok((l_indet, r_indet))
    }

    pub fn zero_massey_product_for(
        &self,
        a: &AdamsElement,
        b: Bidegree,
        c: &AdamsElement,
    ) -> Result<MasseyProduct, String> {
        let (l_ind, r_ind) = self.compute_indeterminacy_of_massey_product(a, b, c)?;
        let bidegree = (a.s() + b.s() + c.s() - 1, a.t() + b.t() + c.t()).into();
        let res_dim = self
            .num_gens_bidegree(bidegree)
            .map(Ok)
            .unwrap_or(Err("massey product resulting group not computed"))?;
        Ok(MasseyProduct::new_ae(
            &(bidegree, FpVector::new(self.prime(), res_dim)).into(),
            l_ind,
            r_ind,
        ))
    }

    /// computes the maximum degree through which multiplication with an adams element is defined
    /// expects the adams generator to be valid
    pub fn multiplication_computed_through_degree(&self, gen: AdamsGenerator) -> Option<Bidegree> {
        self.multiplication_matrices.get(&gen).map(|kv| kv.0)
    }

    /// returns maximum degree such that all multiplications with generators living in multiplier_deg
    /// are defined through that degree. Assumes multiplier_deg <= self.max_deg()
    /// returns None if any prerequisites are not computed
    pub fn multiplication_completely_computed_through_degree(
        &self,
        multiplier_deg: Bidegree,
    ) -> Option<Bidegree> {
        let dim = match self.num_gens_bidegree(multiplier_deg) {
            Some(n) => n,
            None => {
                // bidegree dimension not even computed
                return None;
            }
        };
        // max possible degree if all multiplications are computed
        let mut res = (
            self.max_s - multiplier_deg.s(),
            self.max_t - multiplier_deg.t(),
        )
            .into();
        for idx in 0..dim {
            let max_for_idx =
                match self.multiplication_computed_through_degree((multiplier_deg, idx).into()) {
                    Some(deg) => deg,
                    None => {
                        // multiplication with this generator not computed at all
                        return None;
                    }
                };
            res = meet(res, max_for_idx);
        }
        Some(res)
    }

    /// compute kernels for left multiplication by the given AdamsElement through the given Bidegree
    pub fn compute_kernels_left_multiplication(
        &self,
        multiplier: &AdamsElement,
        max_degree_kernels: Bidegree,
    ) -> Result<(Bidegree, DashMap<Bidegree, Subspace>), String> {
        eprintln!(
            "compute_kernels_left_multiplication({}, {})",
            multiplier, max_degree_kernels
        );
        let deg1 = multiplier.degree();
        let hm = DashMap::new();
        let max_mult_with_deg = match self.multiplication_completely_computed_through_degree(deg1) {
            Some(md) => md,
            None => {
                return Err(format!(
                    "Multiplication not completely computed for bidegree {}",
                    deg1
                ));
            }
        };
        let max_deg_for_kernels = meet(max_degree_kernels, max_mult_with_deg);
        for deg2 in max_deg_for_kernels.iter_s_t() {
            let deg3 = deg1 + deg2;
            let dim2 = match self.num_gens_bidegree(deg2) {
                Some(n) => n,
                None => {
                    // not computed. this shouldn't happen
                    return Err(format!("Trying to compute kernel for {}.\nExpected multiply with degree {} to be computed", multiplier, deg2));
                }
            };
            if dim2 == 0 {
                continue; // no nonzero vectors, kernel is trivial
            }
            let dim3 = match self.num_gens_bidegree(deg3) {
                Some(n) => n,
                None => {
                    // not computed. this shouldn't happen
                    return Err(format!("Trying to compute kernel for {}.\nExpected product degree {} to be computed", multiplier, deg3));
                }
            };
            if dim3 == 0 {
                // kernel is everything
                // add and skip
                hm.insert(deg2, Subspace::entire_space(self.prime(), dim2));
                continue;
            }
            eprintln!("computing nontrivial kernel for deg: {}", deg2);
            let lmul_v1 = match self.left_multiplication_by(deg1, multiplier.vec(), deg2) {
                Ok(m) => m,
                Err(_) => {
                    continue;
                }
            };
            let (aug_start, mut lmul_v1_aug) =
                Matrix::augmented_from_vec(self.prime(), &lmul_v1.to_vec());
            lmul_v1_aug.row_reduce();
            let kernel_lmul_v1 = lmul_v1_aug.compute_kernel(aug_start);
            if kernel_lmul_v1.dimension() == 0 {
                // kernel trival
                continue; // skip
            }
            hm.insert(deg2, kernel_lmul_v1);
        }
        Ok((max_deg_for_kernels, hm))
    }

    /// computes kernels for right multiplication by the given Adams element through the given bidegree
    /// redirects to compute_kernels_left_multiplication right now. TODO
    pub fn compute_kernels_right_multiplication(
        &self,
        multiplier: &AdamsElement,
        max_degree_kernels: Bidegree,
    ) -> Result<(Bidegree, DashMap<Bidegree, Subspace>), String> {
        eprintln!(
            "compute_kernels_right_multiplication({}, {})",
            multiplier, max_degree_kernels
        );
        self.compute_kernels_left_multiplication(multiplier, max_degree_kernels)
    }

    /// computes kernels for left multiplication by all Adams elements through
    /// bidegree max_degree_multiplier with kernels computed through degree
    /// max_degree_kernel.
    /// returns hashmap from Adams elements to pairs of a bidegree representing the maximum degree through which
    /// kernels are computed and a hashmap with kernels by Bidegree
    /// if an in range pair of (nonzero) adams element and bidegree is not in the hashmaps,
    /// it's because the kernel is 0
    pub fn compute_all_kernels_left_multiplication(
        &self,
        max_degree_multiplier: Bidegree,
        max_degree_kernels: Bidegree,
    ) -> Result<DashMap<AdamsElement, (Bidegree, DashMap<Bidegree, Subspace>)>, String> {
        let kernels: DashMap<AdamsElement, (Bidegree, DashMap<Bidegree, Subspace>)> =
            DashMap::new();
        for deg1 in max_degree_multiplier.iter_s_t() {
            let dim1 = match self.num_gens_bidegree(deg1) {
                Some(n) => n,
                None => {
                    return Err(format!("Bidegree {} not computed", deg1));
                }
            };
            if dim1 == 0 {
                continue;
            }
            for v in AllVectorsIterator::new_whole_space(self.prime(), dim1) {
                let ae = AdamsElement::from((deg1, v));
                let kers = self.compute_kernels_left_multiplication(&ae, max_degree_kernels)?;

                kernels.insert(ae, kers);
            }
        }
        Ok(kernels)
    }
    /// computes kernels for right multiplication by all Adams elements through
    /// bidegree max_degree_multiplier with kernels computed through degree
    /// max_degree_kernel
    ///
    /// currently implemented by just calling the compute_kernels_left_multiplication method
    pub fn compute_all_kernels_right_multiplication(
        &self,
        max_degree_multiplier: Bidegree,
        max_degree_kernels: Bidegree,
    ) -> Result<DashMap<AdamsElement, (Bidegree, DashMap<Bidegree, Subspace>)>, String> {
        // TODO right now we're going to assume multiplication is commutative, since we're working with
        // Adams SS for sphere at p=2.
        self.compute_all_kernels_left_multiplication(max_degree_multiplier, max_degree_kernels)
    }

    /// Write a new function to compute the massey products <a,b,c> given
    /// b and c for a less than max_deg_a.
    ///
    /// Should only compute one homotopy. Assumes bc=0.
    ///
    /// Returns the maximum a degree through which Massey products were computed, as well as
    /// a Vector of triples (a, representative, subspace) for which the massey product does not
    /// contain 0. Any possible massey product in the range indicated by the bidegree which is not
    /// recorded contains 0.
    ///
    /// takes the kernels for right multiplication by b as an argument
    pub fn compute_massey_prods_for_pair(
        &self,
        kernels_mul_b: &(Bidegree, DashMap<Bidegree, Subspace>),
        max_deg_a: Bidegree,
        b: &AdamsElement,
        c: &AdamsElement,
    ) -> (Bidegree, Vec<(AdamsElement, MasseyProduct)>) {
        eprintln!(
            "compute_massey_prods_for_pair(kernels, {}, {}, {})",
            max_deg_a, b, c
        );
        let mut ans = vec![];
        let (kernels_max_deg, ker_map) = kernels_mul_b;
        let b_deg = b.degree();
        let c_deg = c.degree();
        let b_c_deg = b_deg + c_deg;
        let b_c_shift = match b_c_deg.try_subtract((1, 0).into()) {
            Some(shift) => shift, // this is the degree difference |<a,b,c>| - |a|
            None => {
                return ((0, 0).into(), ans); // return empty data structure.
                                             // this only happens if b and c have s degree 0, and therefore are either 0 or 1,
                                             // which have no interesting massey products
            }
        };
        // the extra 1 in s degree is due to massey product living in degree deg_a + deg_b + deg_c - (1,0)
        let complement = match (self.max_deg() + (1, 0).into()).try_subtract(b_c_deg) {
            Some(cmpl) => cmpl,
            None => {
                return ((0, 0).into(), ans); // return empty data structure, since there are no valid a's to multiply with anyway
            }
        };
        // complement represents maximum possible a degree we can compute with
        let max_a = max_deg_a.meet(complement).meet(*kernels_max_deg); // intersect the ranges
        eprintln!("determined max_a: (s,t)=({},{})", max_a.s(), max_a.t());

        let (max_s1, max_t1) = max_a.into();
        let (s2, t2, v2) = b.into();
        let (s3, t3, v3) = c.into();
        // largest degree increase from c to <a,b,c>
        let (max_shift_s, max_shift_t) = (max_s1 + s2 - 1, max_t1 + t2);
        eprintln!("max_shift: (s,t)=({},{})", max_shift_s, max_shift_t);
        //let shift_n = shift_t-shift_s as i32;
        // largest total degree of <a,b,c>
        let (max_tot_s, max_tot_t) = (max_shift_s + s3, max_shift_t + t3);
        eprintln!("max_tot: (s,t)=({},{})", max_tot_s, max_tot_t);

        //let tot_n = tot_t-tot_s as i32;

        // for now we'll just compute the resolutions for b and c
        // this can be computed in terms of cached data in principle, and it should be faster,
        // but it requires a change to Hood's library
        eprint!("lifting {} to resolution homomorphism...", b);
        let res_hom_2 = self.adams_elt_to_resoln_hom(b);
        res_hom_2.extend(max_shift_s, max_shift_t); // TODO concurrentify
        eprintln!(" done.");
        eprint!("lifting {} to resolution homomorphism...", c);
        let res_hom_3 = self.adams_elt_to_resoln_hom(c);
        res_hom_3.extend(max_tot_s, max_tot_t); // TODO concurrentify
        eprintln!(" done.");
        // workout ordering
        eprint!("constructing {} o {}...", b, c);
        let composite = ResolutionHomomorphism::new(
            "b o c".into(),
            self.resolution(),
            self.resolution(),
            s2 + s3,
            t2 + t3,
        );
        for s in 0..=max_s1 {
            // composite map
            let source = self.resolution().module(s + s2 + s3);
            let map_c = res_hom_3.get_map(s + s2 + s3);
            let intermediate = self.resolution().module(s + s2);
            let map_b = res_hom_2.get_map(s + s2);
            let target = self.resolution.module(s);
            for t in 0..=max_t1 {
                // compute composite matrix
                // c goes from deg s+s2+s3, t+t2+t3 to
                // s+s2, t+t2
                let num_source_gens = source.number_of_gens_in_degree(t + t2 + t3);
                // b goes from deg s+s2, t+t2
                // to s, t
                let intermediate_dim = intermediate.dimension(t + t2);
                let target_dim = target.dimension(t);
                let mut composite_matrix = Matrix::new(self.prime(), target_dim, num_source_gens);
                let mut intermediate = FpVector::new(self.prime(), intermediate_dim);
                for i in 0..num_source_gens {
                    map_c.apply_to_generator(&mut intermediate, 1, t + t2 + t3, i);
                    map_b.apply(
                        composite_matrix[i].as_slice_mut(),
                        1,
                        t + t2,
                        intermediate.as_slice(),
                    );
                }
                composite.extend_step(s + s2 + s3, t + t2 + t3, Some(&composite_matrix));
            }
        }
        eprintln!("done.");
        eprint!("constructing 0...");
        let zero = ResolutionHomomorphism::new(
            "zero".into(),
            self.resolution(),
            self.resolution(),
            s2 + s3,
            t2 + t3,
        );
        for s in 0..=max_s1 {
            // composite map
            let source = self.resolution().module(s + s2 + s3);
            let target = self.resolution.module(s);
            for t in 0..=max_t1 {
                // compute composite matrix
                // c goes from deg s+s2+s3, t+t2+t3 to
                // s+s2, t+t2
                let num_source_gens = source.number_of_gens_in_degree(t + t2 + t3);
                // b goes from deg s+s2, t+t2
                // to s, t
                let target_dim = target.dimension(t);
                let zero_matrix = Matrix::new(self.prime(), target_dim, num_source_gens);
                composite.extend_step(s + s2 + s3, t + t2 + t3, Some(&zero_matrix));
            }
        }
        eprintln!("done.AsMut");
        eprint!("computing nullhomotopy of {} o {}...", b, c);
        let homotopy = ChainHomotopy::new(
            Arc::new(composite),
            Arc::new(zero), // TODO
                            /*&*self.resolution,
                            &*self.resolution,
                            s2+s3,
                            t2+t3,
                            |source_s, source_t, idx, row| {
                                let mid_s = source_s - s3;

                                // ought to represent res_hom_2 \circ res_hom_3, which makes sense
                                res_hom_3.get_map(source_s)
                                    .compose(res_hom_2.get_map(mid_s))
                                    .apply_to_basis_element(row.as_slice_mut(), 1, source_t, idx);
                            }
                            */
        );

        for s in b_c_deg.s()..=max_tot_s {
            eprint!(" extend htpy to ({},{})", s, max_tot_t);
            homotopy.extend(s, max_tot_t);
            // need the loop b/c extend uses stem degree
            // i.e. extends row s to t = max_n + s
            // so the first few rows end up too short
        }
        eprintln!(" done.");

        eprintln!("extracting massey products from homotopy...");
        // extract representatives for massey products from homotopy
        for a_deg in max_deg_a.iter_s_t() {
            if a_deg.n() < 0 {
                continue;
            }
            let ker_b_dim_a = match ker_map.get(&a_deg) {
                Some(subsp) => {
                    if subsp.dimension() == 0 {
                        eprintln!("no vectors in kernel here, done. But kernel is recorded, which shouldn't happen.");
                        continue; // kernel is trivial, nothing interesting here.
                    }
                    subsp
                }
                None => {
                    //eprintln!("no vectors in kernel here, done.");
                    continue; // means that the kernel is trivial, nothing interesting here
                }
            };
            let (s1, t1) = a_deg.into();
            let tot_deg = a_deg + b_c_shift;
            let (tot_s, tot_t) = tot_deg.into();
            let target_dim = match self.num_gens_bidegree(tot_deg) {
                Some(n) => n,
                None => {
                    eprintln!("Error, expected dimension of target for massey product <{},{},{}> to be computed", a_deg, b, c);
                    continue; // ignore TODO
                }
            };
            if target_dim == 0 {
                //eprintln!("target empty, done.");
                continue;
            }
            eprint!("for deg {}... ", a_deg);
            let htpy_map = homotopy.homotopy(tot_s);
            let offset_a = self.resolution.module(s1).generator_offset(t1, t1, 0); // where do generators
                                                                                   // start in the basis after all the products and what
            for vec_a in AllVectorsIterator::new(&ker_b_dim_a) {
                if vec_a.is_zero() {
                    continue; // trivial massey products
                }
                // now htpy_map
                // goes from the free module in degree tot_s to the one in degree a_deg
                // need to compose with the map given by vec_a from a_deg to F2
                // and then read off the values of the composite map
                // on the generators in degree tot_s, tot_t
                let mut answer = vec![0; target_dim];
                let mut nonzero = false;
                eprintln!(" computing for {}...", vec_a);
                for (i, ans) in answer.iter_mut().enumerate().take(target_dim) {
                    let output = htpy_map.output(tot_t, i);
                    eprintln!(
                        "output of htpy for ({}, {}) index {} = {}",
                        tot_s, tot_t, i, output
                    );
                    for (k, entry) in vec_a.iter().enumerate() {
                        if entry != 0 {
                            //answer[i] += entry * output.entry(self.resolution.module(s1).generator_offset(t1,t1,k));
                            *ans += entry * output.entry(offset_a + k);
                        }
                    }
                    if *ans != 0 {
                        nonzero = true;
                    }
                }
                eprintln!(" rep for <{},-,->={:?}", vec_a, answer);
                if nonzero {
                    let massey_rep = FpVector::from_slice(self.prime(), &answer);
                    eprintln!(" nonzero rep for <{},-,->={}", vec_a, massey_rep);
                    let ae1 = (a_deg, &vec_a).into();
                    let indets = match self.compute_indeterminacy_of_massey_product(&ae1, b_deg, c)
                    {
                        Ok(indets) => indets,
                        Err(reason) => {
                            eprintln!(
                                "< ({}, {}, {}), ({}, {}, {}), ({}, {}, {}) > =
                                ({}, {}, {}) + {{??}} could not compute indeterminacy because {}",
                                s1,
                                t1,
                                vec_a,
                                s2,
                                t2,
                                v2,
                                s3,
                                t3,
                                v3,
                                tot_s,
                                tot_t,
                                massey_rep,
                                reason,
                            );
                            // hopefully this doesn't happen
                            continue; // printed out, keep on going
                        }
                    };
                    let massey_prod =
                        MasseyProduct::new(tot_s, tot_t, massey_rep, indets.0, indets.1);
                    if massey_prod.contains_zero() {
                        continue; // massey product is trivial, ignore it
                    }
                    ans.push((ae1, massey_prod))
                }
            }
            eprintln!("done.");
        }
        (max_a, ans)
    }

    /*
    pub fn compute_massey_products(&self, max_massey: Bidegree) {

    }
    */

    /*
    /// compute all massey products of massey-productable triples (a,b,c)
    /// all of whose bidegrees are less than max_massey
    pub fn brute_force_compute_all_massey_products(&self, max_massey: Bidegree) {
        let mut zero_massey_output = match File::create("zero-massey-prods.txt") {
            Err(error) => { eprintln!("Could not open 'zero-massey-prods.txt' for writing: {}", error); return; }
            Ok(file) => file
        };
        let p = self.prime();
        let (max_mass_s, max_mass_t) = max_massey.into();
        // first identify kernels of left multiplication in this range
        let mut kernels: DashMap<(Bidegree,Bidegree), DashMap<FpVector,Subspace>> = DashMap::new();
        for s1 in 1..max_mass_s {
            for t1 in s1 as i32..max_mass_t {
                let dim1 = match self.num_gens(s1, t1) {
                    Some(n) => n,
                    None => { continue; } // not computed. this shouldn't happen
                };
                if dim1 == 0 {
                    continue; // no nonzero vectors
                }
                for v1 in AllVectorsIterator::new_whole_space(p, dim1) {
                    if v1.is_zero() { // no need to consider the 0 vector
                        continue;
                    }
                    // might go out of bounds if max_mass_s, max_mass_t > 0.5 max_s, max_t
                    // TODO
                    for s2 in 1..max_mass_s {
                        for t2 in s2 as i32..max_mass_t {
                            let (s3, t3) = (s1+s2, t1+t2);
                            let dim2 = match self.num_gens(s2, t2) {
                                Some(n) => n,
                                None => { continue; } // not computed. this shouldn't happen
                            };
                            let _dim3 = match self.num_gens(s3, t3) {
                                Some(n) => n,
                                None => { continue; } // not computed. this shouldn't happen
                            };
                            if dim2 == 0 {
                                continue; // no nonzero vectors
                            }
                            let lmul_v1 = match self.left_multiplication_by((s1, t1).into(), &v1, (s2, t2).into()) {
                                Ok(m) => m,
                                Err(_) => {
                                    continue;
                                }
                            };
                            let (aug_start, mut lmul_v1_aug) = Matrix::augmented_from_vec(p, &lmul_v1.to_vec());
                            lmul_v1_aug.row_reduce();
                            let kernel_lmul_v1 = lmul_v1_aug.compute_kernel(aug_start);
                            if kernel_lmul_v1.dimension() == 0 {
                                // kernel trival
                                continue; // skip
                            }
                            let bidegree_pair = ((s1,t1).into(),(s2,t2).into());
                            match kernels.get_mut(&bidegree_pair) {
                                Some(hm) => {
                                    hm.insert(v1.clone(),kernel_lmul_v1.clone());
                                },
                                None => {
                                    let mut new_hm = DashMap::new();
                                    new_hm.insert(v1.clone(), kernel_lmul_v1.clone());
                                    kernels.insert(bidegree_pair, new_hm);
                                }
                            }
                            /*
                            for v2 in AllVectorsIterator::new(&kernel_lmul_v1) {
                                if v2.is_zero() {
                                    continue;
                                }
                                println!("({},{},{})*({},{},{}) = ({},{},{})", s1, t1, v1, s2, t2, v2, s3, t3, format!("0_{}", dim3));
                            }
                            */
                        }
                    }
                }
            }
        }
        // stores interesting (no element zero, and lands in nontrivial degree for now) massey-productable triples
        let mut triples: Vec<(AdamsElement, AdamsElement, AdamsElement)> = Vec::new();
        for s1 in 1..max_mass_s {
            for t1 in s1 as i32..max_mass_t {
                let deg1 = (s1, t1).into();
                for s2 in 1..max_mass_s {
                    for t2 in s2 as i32..max_mass_t {
                        let deg2 = (s2, t2).into();
                        let bideg_pr_l = (deg1, deg2);
                        let hm_kers = match kernels.get(&bideg_pr_l) {
                            Some(hm_kers) => {
                                hm_kers
                            },
                            None => { continue; } // no interesting vectors/kernels in this bidegree pair
                        };
                        for (v1, ker_v1) in hm_kers.iter() {
                            for v2 in AllVectorsIterator::new(ker_v1) {
                                if v2.is_zero() {
                                    continue; // skip the zero vector
                                }
                                // now iterate over s3, t3
                                for s3 in 1..max_mass_s {
                                    for t3 in s3 as i32..max_mass_t {
                                        let deg3 = (s3, t3).into();
                                        let bideg_pr_r = (deg2, deg3);
                                        let final_bideg = (s1+s2+s3-1, t1+t2+t3).into();
                                        match self.num_gens_bidegree(final_bideg) {
                                            Some(n) => {
                                                // computed bidegree
                                                if n==0 { // but dimension 0, no interesting massey products
                                                    continue;
                                                }
                                            },
                                            None => {
                                                // uncomputed bidegree, skip
                                                continue;
                                            }
                                        }
                                        let hm_kers_2 = match kernels.get(&bideg_pr_r) {
                                            Some(hm_kers_2) => { hm_kers_2 },
                                            None => { continue; } // no interesting vectors/kernels in this
                                            // bidegree pair
                                        };
                                        let ker_v2 = match hm_kers_2.get(&v2) {
                                            Some(ker_v2) => { ker_v2 },
                                            None => { continue; } // v2 doesn't have an interesting kernel here
                                        };
                                        for v3 in AllVectorsIterator::new(ker_v2) {
                                            if v3.is_zero() {
                                                continue;
                                            }
                                            triples.push(((s1,t1,v1.clone()).into(), (s2,t2,v2.clone()).into(), (s3, t3, v3.clone()).into()));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (ae1, ae2, ae3) in &triples {
            let (s1,t1,v1) = ae1.into();
            let (s2,t2,v2) = ae2.into();
            let (s3,t3,v3) = ae3.into();
            let (shift_s,shift_t) = (s1+s2-1, t1+t2);
            let shift_n = shift_t-shift_s as i32;
            let (tot_s, tot_t) = (shift_s+s3, shift_t+t3);
            let tot_n = tot_t-tot_s as i32;
            let target_dim = match self.num_gens(tot_s, tot_t) {
                Some(n) => n,
                None => { continue; }
            };

            let res_hom_2 = self.adams_elt_to_resoln_hom(ae2);
            res_hom_2.extend_through_stem(shift_s, shift_n);
            let res_hom_3 = self.adams_elt_to_resoln_hom(ae3);
            res_hom_3.extend_through_stem(tot_s, tot_n);

            // TODO
            let homotopy = ChainHomotopy::new(
                &*self.resolution,
                &*self.resolution,
                s2+s3,
                t2+t3,
                |source_s, source_t, idx, row| {
                    let mid_s = source_s - s3;

                    res_hom_3.get_map(source_s)
                        .compose(res_hom_2.get_map(mid_s))
                        .apply_to_basis_element(row.as_slice_mut(), 1, source_t, idx);
                }
            );

            homotopy.extend(tot_s, tot_t);
            let last = homotopy.homotopy(tot_s);
            let mut answer = vec![0; target_dim];
            let mut nonzero = false;
            for i in 0..target_dim {
                let output = last.output(tot_t, i);
                for (k, entry) in v1.iter().enumerate() {
                    if entry != 0 {
                        answer[i] += entry * output.entry(k); // TODO: might need an offset here

                    }
                }
                if answer[i]!=0 {
                    nonzero=true;
                }
            }
            //for

            if nonzero {
                let massey_rep = FpVector::from_slice(v1.prime(), &answer);
                let indet = match self.compute_indeterminacy_of_massey_product(ae1, (s2, t2).into(), ae3) {
                    Ok((l_sub,r_sub)) => utils::subspace_sum(&l_sub, &r_sub),
                    Err(reason) => {
                        println!("< ({}, {}, {}), ({}, {}, {}), ({}, {}, {}) > = ({}, {}, {}) + {:?}",
                            s1, t1, v1,
                            s2, t2, v2,
                            s3, t3, v3,
                            tot_s, tot_t, massey_rep,
                            format!("{} could not compute indeterminacy because {}", "{??}", reason)
                            );
                        // hopefully this doesn't happen
                        continue; // printed out, keep on going
                    }
                };
                print!("< ({}, {}, {}), ({}, {}, {}), ({}, {}, {}) > = ({}, {}, {}) + {:?}",
                    s1, t1, v1,
                    s2, t2, v2,
                    s3, t3, v3,
                    tot_s, tot_t, massey_rep,
                    indet
                );
                if indet.contains(massey_rep.as_slice()) {
                    println!(" = 0 ")
                } else {
                    println!();
                }
            } else {
                let _ = writeln!(zero_massey_output, "< ({}, {}, {}), ({}, {}, {}), ({}, {}, {}) > = 0 + did not compute indeterminacy",
                    s1, t1, v1,
                    s2, t2, v2,
                    s3, t3, v3,
                );
            }
        }
        println!("{} total triples", triples.len());

    }
    */
}
