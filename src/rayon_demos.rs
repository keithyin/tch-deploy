use rayon;

fn rayon_thread_pool() {
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(10).build().unwrap();

    let do_iter = || {
        pool.install(|| {
            println!("A ");
            println!("B ");
        })
    };

    rayon::join(|| do_iter(), || do_iter());

}

#[cfg(test)]
mod test {
    use crate::rayon_demos::rayon_thread_pool;


    #[test]
    fn test_tp() {
        rayon_thread_pool();
    }
}