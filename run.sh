#!/bin/bash

###### Default Values

# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=tmp -weight=5.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True

###### END

### no control vs guide words only vs glove guidance
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -top_p=0.9 -n_generated_sentences=90 -guide= 
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=5.0 -top_p=0.9 -n_generated_sentences=90 -only_max=True
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=10.0 -top_p=0.9 -n_generated_sentences=90 -only_max=True
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=20.0 -top_p=0.9 -n_generated_sentences=90 -only_max=True
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=5.0 -top_p=0.9 -n_generated_sentences=90 
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=10.0 -top_p=0.9 -n_generated_sentences=90 
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=20.0 -top_p=0.9 -n_generated_sentences=90 

### evaluating shift strength
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=5.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=10.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=15.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=20.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=25.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=guide_vs_no_guide_beams -weight=30.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True

### comparing different unordered modes
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=comparing_modes -weight=5.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True
# python main_DBS.py -mode='max' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=comparing_modes -weight=5.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True
# python main_DBS.py -mode='random' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=comparing_modes -weight=5.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True
# python main_DBS.py -mode='all' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=comparing_modes -weight=5.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True

### comparing different decoding methods
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=decoding_methods -weight=5.0 -top_p=0.9 -n_generated_sentences=90 -do_guarantee=True
# python main_DBS.py -mode='next' -file_name=/data/50_keywordsets_eval/word_sets.txt -results_subfolder=decoding_methods -weight=5.0 -top_p=0.0 -n_generated_sentences=90 -do_guarantee=True

### ROC
# python main_DBS.py -mode='max' -file_name=/data/ROC/ROCStories_20_storylines_500_0.txt -results_subfolder=final4_ -weight=5.0 -top_p=0.9 -n_generated_sentences=-7 -n_beams=4 -do_guarantee=True

### Keyword to Article
python main_DBS.py -mode='max' -file_name=/data/keyword_to_articles -results_subfolder=tmp -key2article=True -weight=5.0 -top_p=0.9 -n_generated_sentences=-15 -n_beams=4 -do_guarantee=True

