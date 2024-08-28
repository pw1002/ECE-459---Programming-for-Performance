PR Title:
Implemented threading (concurrent hashmaps) in parsing log files

Summary:
For this implementation (when single-maps is false or not specified), I added the breaking up of the log file into segments
and processed them in seperate threads. I had implemented a concurrent hashmap in the form of a dashmap and wrapped them in Arc
to allow concurrency and sharing. I still kept the all_tokens_list the same as it was specified in the lab document to only implement
the concurrency for the dbl and trpl hashmaps. I did not need to combine the results of dbl and trpl as they are already shared, but I 
still did for the all_tokens_list.

Tech Details:
The most noteworthy thing about this implementation is how I split up the segments and how I passed in these 
segments into the functions for parsing. Though I already talked about this in the seperate_maps.md file so I will
just summarize it here. I did it the same way for both the single map and the concurrent map implementation.

I simply passed in the next segment's first line as the lookahead to the last line in the previous segment and at the 
same time I passed in the last two tokens of the last line in the previous segment to the new segment if it exists. This 
way I could double count the 2 and 3-grams that are split from the segements.

The other thing I implemented in the concurrent hashmap version was the dashmap. This was a crate I found on the internet. I wrapped these in
Arc to help implement the concurrency part. Other than that there was not much difference. I had to change some variable typing and created new 
methods to take in Arc<Dashmap<>> intead of the hashmap. Then I returned the Dashmap at the very end in the form of a hashmap.

Correctness:
To test my code for correctness, I ran the commands in the readme file with the original implementation of dictionary_builder() with 
no threading or anything that I added of my own. I save the output of this and compared it with my own on several different outputs for 
different commands found in the readme file. Parameters I changed here were the number of threads (since single-maps should always be toggled). 
I found that adding threads still yielded the same output (same dictionaries and same counts) as the original. I also tested and ran the unit test 
that was given with different number of threads, which passed so it should be correct. It is also worth to note that I ran the unit test with the 
linux setting as the readme commands also covered the spark setting so both should be correct.

Performance: 
The way I tested for performance was running the commands with different threads and also the unit test with different threads, and measuring the times
that was measured. It was very varied sometimes but on average it was not faster than the original implementation. Sometimes was around the same or lower 
most of the time. 