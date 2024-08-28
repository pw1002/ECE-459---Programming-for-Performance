use urlencoding::encode;

use curl::easy::{Easy2, Handler, WriteError};
use curl::multi::{Easy2Handle, Multi};
use std::collections::VecDeque;
use std::time::Duration;
use std::str;
use std::sync::atomic::{AtomicI32, Ordering};

use crate::Packages;

struct Collector(Box<String>);
impl Handler for Collector {
    fn write(&mut self, data: &[u8]) -> Result<usize, WriteError> {
        (*self.0).push_str(str::from_utf8(&data.to_vec()).unwrap());
        Ok(data.len())
    }
}

const DEFAULT_SERVER : &str = "ece459.patricklam.ca:4590";
impl Drop for Packages {
    fn drop(&mut self) {
        self.execute()
    }
}

static EASYKEY_COUNTER: AtomicI32 = AtomicI32::new(0);

pub struct AsyncState {
    server : String,
    queue : Vec<Easy2Handle<Collector>>,
    multi : Multi,
    md5sums : VecDeque<String>,
}

impl AsyncState {
    pub fn new() -> AsyncState {
        AsyncState {
            server : String::from(DEFAULT_SERVER),
            queue : Vec::new(),
            multi : Multi::new(),
            md5sums: VecDeque::new(),
        }
    }
}

impl Packages {
    pub fn set_server(&mut self, new_server:&str) {
        self.async_state.server = String::from(new_server);
    }

    /// Retrieves the version number of pkg and calls enq_verify_with_version with that version number.
    pub fn enq_verify(&mut self, pkg:&str) {
        let version = self.get_available_debver(pkg);
        match version {
            None => { println!("Error: package {} not defined.", pkg); return },
            Some(v) => { 
                let vs = &v.to_string();
                self.enq_verify_with_version(pkg, vs); 
            }
        };
    }

    /// Enqueues a request for the provided version/package information. Stores any needed state to async_state so that execute() can handle the results and print out needed output.
    pub fn enq_verify_with_version(&mut self, pkg:&str, version:&str) {
        let url = format!("http://ece459.patricklam.ca:4590/rest/v1/checksums/{}/{}", pkg, version);

        let mut easy = Easy2::new(Collector(Box::new(String::new())));

        easy.url(&url).unwrap();
        easy.get(true).unwrap();

        println!("queueing request {}", url);

        // add to multi
        match self.async_state.multi.add2(easy) {
            Ok(easy_handle) => self.async_state.queue.push(easy_handle),
            Err(err) => {
                println!("Failed to add Easy2 handle: {}", err);
                return;
            }
        }

        //save in async
        self.async_state.md5sums.push_back(self.get_md5sum(pkg).unwrap().to_string());
    }

    /// Asks curl to perform all enqueued requests. For requests that succeed with response code 200, compares received MD5sum with local MD5sum (perhaps stored earlier). For requests that fail with 400+, prints error message.
    pub fn execute(&mut self) {
        let queue_ref = &mut self.async_state.queue; //&Vec<Easy2Handle<Collector>> type
        let multi_ref = &self.async_state.multi;

        while multi_ref.perform().unwrap() > 0 {
            multi_ref.wait(&mut [], Duration::from_secs(30)).unwrap();
        }

        for eh in queue_ref.drain(..) {
            let mut handler_after:Easy2<Collector> = multi_ref.remove2(eh).unwrap();
            let res_code = handler_after.response_code().unwrap();
            if res_code == 200 {
                let mut same_md5sum: bool = false;
                let server_md5: String = *handler_after.get_ref().0.clone();

                let url_string = handler_after.effective_url().unwrap();
                // parse the URL
                if let Some(url_string) = url_string {
                    if let Some(last_slash) = url_string.rfind('/') {
                        if let Some(second_last_slash) = url_string[..last_slash].rfind('/') {
                            let pkg = &url_string[second_last_slash + 1..last_slash];
                            let local_md5sum: String = self.async_state.md5sums.pop_front().unwrap();
                            if server_md5 == local_md5sum {
                                same_md5sum = true;
                            }
                            println!("verifying {}, matches: {:?}", pkg, same_md5sum);
                        }
                    }
                }
            }
            else {
                let url_string = handler_after.effective_url().unwrap();
                // parse the URL
                if let Some(url_string) = url_string {
                    if let Some(last_slash) = url_string.rfind('/') {
                        if let Some(second_last_slash) = url_string[..last_slash].rfind('/') {
                            let package_name = &url_string[second_last_slash + 1..last_slash];
                            let version = &url_string[last_slash + 1..];
            
                            println!("got error {} on request for package {} version {}", res_code, package_name, version);

                        }
                    }
                }
            }
        }
    }
}
