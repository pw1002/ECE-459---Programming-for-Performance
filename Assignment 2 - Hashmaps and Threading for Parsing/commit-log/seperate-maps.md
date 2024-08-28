PR Title:
Implemented threading (non-concurrent hashmaps) in parsing log files

Summary:
For this implementation (when single-maps is true), I added the breaking up of the log file into segments
and processed them in seperate threads. I made new hashmaps for each thread and passed in those hashmaps to 
the threads. Then returned the hashmaps from each thread and combined these
hashmaps to make up the original dbl and trpl and all_token_list hashmaps and vectors.

Tech Details:
The most noteworthy thing about this implementation is how I split up the segments and how I passed in these 
segments into the functions for parsing. I feel like it was done in a very specific way. To start I created a new
struct called FileSegment which kept track of segments in the log file. It kept track of things like the start and end 
of the segment, the total number of lines in the file, and also the next line if it exists. 

The reason this was done was because I had to account for the double count that was present in the test cases. This means 
that at the end of each segment, I actually need to pass in the first line of the next segment as the lookahead line to 
take into account the 2 and 3-grams that span different segments. Similarly, I also needed to get the last 2 tokens of 
the previous segment (prev1 and prev2) and give it to the first line of the next segment. Obviously this is excluding the 
last segment and first segment respectively.

This way I would be able to account for the double count. Once on the last line of the previous segement and another time on 
the next segment. I was also able to do this because of the functionality of the process_dictionary_builder_line() function. 
This function would add to the end of line the first two tokens from lookahead_line, and return the first 2 tokens on this line.

Correctness:
To test my code for correctness, I ran the commands in the readme file with the original implementation of dictionary_builder() with 
no threading or anything that I added of my own. I save the output of this and compared it with my own on several different outputs for 
different commands found in the readme file. Parameters I changed here were the number of threads (since single-maps should always be toggled). 
I found that adding threads still yielded the same output (same dictionaries and same counts) as the original. I also tested and ran the unit test 
that was given with different number of threads, which passed so it should be correct. It is also worth to note that I ran the unit test with the 
linux setting as the readme commands also covered the spark setting so both should be correct.

Performance: 
The way I tested for performance was running the commands with different threads and also the unit test with different threads, and measuring the times
that was measured. It was very varied sometimes but on average it was faster than the original implementation. Especially, the more threads I used, the 
faster it would typically be. However, this was much more evident in longer log files as the shorter log files sometimes provided outliers as it was much 
quicker to parse 9 lines of logs using the original implementation than sharing and locking and unlocking. But overall, this implementation works and is much 
faster for larger log files