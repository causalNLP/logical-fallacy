import os
import sys
path = os.path.abspath(os.path.join(__file__ ,"../../"))
# path = os.path.abspath(os.pardir)
sys.path.append(path) # sys.path.append("../..")

from file_paths import PreprocessFilePaths
FP = PreprocessFilePaths()
from efficiency.log import show_var

class Constants:
    raw_data_folder = FP.preprocess_data_folder
    raw_data_file_pattern = FP.raw_data_quiz_file_pattern

    other_sent_csv = '../'

    data_folder = FP.preprocess_data_folder
    fallacy_list_url = FP.fallacy_list_url
    fallacy_list_html = FP.fallacy_list_html
    fallacy_list_csv = FP.fallacy_list_csv
    fallacy_list_txt = FP.fallacy_list_txt
    file_bad_ques_pref = FP.file_bad_ques_pref
    file_bad_ques_suff = FP.file_bad_ques_suff
    file_bad_defi_only = FP.file_bad_defi_only
    file_bad_incl_terms = FP.file_bad_incl_terms

    clean_data_csv = FP.clean_data_csv
    deleted_data_csv = FP.deleted_data_csv
    data_csv_unaligned = FP.data_csv_unaligned
    clean_data_csv_normed = FP.clean_data_csv_normed
    data_final_format_csv = FP.data_final_format_csv
    data_manual_csv = FP.raw_data_manual
    data_logicallyfallacious_csv = FP.raw_data_logicallyfallacious_csv
    data_propaganda = FP.raw_data_propaganda

    output_folder = FP.final_data_folder
    data_merged_csv = FP.final_edu_data
    # data_merged_csv = output_folder + 'sentence_logic_propa.csv'

    min_sent_len = 5
    from efficiency.log import fread, read_csv
    bad_ques_pref = fread(file_bad_ques_pref, if_strip=True, delete_empty=True)
    bad_ques_suff = fread(file_bad_ques_suff)
    bad_ques_suff = [i.rstrip('\n') for i in bad_ques_pref if i.rstrip('\n')]
    bad_incl_terms = fread(file_bad_incl_terms, if_strip=True, delete_empty=True)

    bad_defi_only = [i['definition'] for i in read_csv(file_bad_defi_only)]

    def __init__(self):
        from glob import glob
        self.raw_data_files = glob(self.raw_data_file_pattern)

        self.fallacy_types_from_wiki = self.get_fallacy_types()
        self.fallacy_types = self.get_cleaned_list_of_fallacies()
        self.contain_which_fallacy = lambda i: \
            [f for f in self.fallacy_types if
             f.lower().replace('-', ' ').replace('/', ' ').replace('  ', ' ')
             in
             i.lower().replace('’', "'").replace('-', ' ').replace('/', ' ').replace('  ', ' ')
                 .replace('appeals to ', 'appeal to ').replace('appealing to ', 'appeal to ')
                 .replace('appeal of ', 'appeal to ')]

        show_var(['len(self.fallacy_types_from_wiki)', 'len(self.fallacy_types)'], joiner=', ')

    def get_cleaned_list_of_fallacies(self):
        from efficiency.function import flatten_list, lstrip_word, rstrip_word
        def _clean_fallacy_name(i):
            i = i.lower()

            safe_single_words = {'circular', 'bandwagon', 'non-sequitur', 'black-or-white', 'incredulity',
                                 'ambiguity', 'anecdotal', 'bulverism', 'concretism', 'contextomy', 'contextotomy',
                                 'either/or', 'equivocation', 'hypostatization', 'name-calling', 'nit-picking',
                                 'novitatis', 'nut-picking', 'pooh-pooh', 'reification', 'slanting', }

            for prefix in ['argumentum ', 'argument from ', 'fallacy of the ', 'fallacy of ', 'the ']:
                after_strip = lstrip_word(i, prefix)
                # if any(j in i for j in {'accident', 'beard'}):
                #     import pdb;pdb.set_trace()
                if (len(lstrip_word(after_strip, 'the ').split()) > 1) or (after_strip in safe_single_words):
                    i = after_strip

            for term in ['fallacy', 'argument', 'claim', 'reasoning']:
                after_strip = rstrip_word(i, ' ' + term)
                if after_strip.lower() not in {'begging the', 'argument from'}:
                    if (len(after_strip.split()) > 1) or (after_strip in safe_single_words):
                        i = after_strip
            return i

        synonyms = {
            'hasty generalization': ['Biased Sample Fallacy', 'overgeneraliz', 'Faulty Generalization',
                                     'Stereotyp', 'It stereotypes.',
                                     'Categorical Claim', 'Sweeping Generaliz', 'Dicto Simpliciter',
                                     'Anecdotal'],
            'insufficient statistics': ['Observational Selection', 'Questionable use of statistics'],
            'equivocation': ['Vagueness', 'Ambiguity'],
            'fallacy of composition': ['Composition Fallacy'],
            'appeal to pity': ['ad misericordiam', ],
            'appeal to authority': ['ethical appeal'],
            'inconsistent comparison': ['Faulty Comparison'],
            'appeal to common sense': ['Appeal to Normality'],
            'appeal to emotion': ['sentimental appeal', 'Emotional Appeal', 'Argument by Emotive Language', 'glittering generalities', 'Loaded Language'],
            'appeal to fear': ['Scare Tactics'],
            'false authority': ['faulty authority', 'False (irrelevant) Authority', 'irrelevant authority',
                                'False/Not Authority', 'inappropriate authority',
                                'Illegitimate Authority', 'Questionable Authority', 'weak Authority'],
            'affirming the consequent': ['Converse Error'],
            'ad hominem': ['name-calling', 'Personal attack', 'Fallacy of Opposition', 'It attacks the person. '],
            'ad populum': ['appeal to popularity', 'Appeal to the Crowd', 'Appeal to Crowd',
                           'Appeal to common belief',
                           'It tells you to like something because a lot of people do.'],
            'argument to moderation': ['Golden Mean Fallacy'],
            'fallacy of the beard': ['Argument of the Beard'],
            'argument from ignorance': ['Fallacy of Ignorance', 'Burden of proof'],
            'black-or-white': ['either/or', 'Black and White', 'Black & White', 'false dilema', 'Fake Dilemma',
                               'Disjunctive Fallacy'],
            'composition fallacy': ['Division Fallacy'],
            'continuum fallacy': ['False Continuum'],
            'magical thinking': ['superstition', ],
            'loaded question': ['Leading Question', ],
            'straw man': ['strawman', 'Straw Person', 'Stawman', 'Strawmam',
                          'It changes the argument to something else. '],
            'bandwagon': ['Band wagon', ],
            'ad ignorantiam': ['Ad Ignorantium', ],
            'guilt by association': ['It says A is true and B is true so A+B must be true.', ],
            'begging the question': ['begging the claim', ],
            'post hoc ergo propter hoc': ['post hoc', ],
            'false analogy': ['faulty analogy', 'bad analogy', 'false or careless analogy', 'Weak analogy'],
            'slipperty slope': ["It leads to something that probably won't happen."],
            'faulty cause/effect': ['false cause and effect', 'False Effect', 'false caus', 'causality', ' caus',
                                    'Causal Fallacy'],
            'correlation implies causation': [],
            'moral equivalence': [],

            'No Fallacy': [],
            'guilt by association and honor by association': ['guilt by association', 'honor by association']
        }
        used = set()
        fallacies = self.fallacy_types_from_wiki
        for standard_name, akas in fallacies.items():
            all_names = [standard_name] + [i for i in akas if i]
            all_names = [_clean_fallacy_name(i) for i in all_names]
            overlap = set(all_names) & set(synonyms)
            new_akas = flatten_list(synonyms[i] for i in overlap)
            new_akas += list(overlap - {standard_name} - {_clean_fallacy_name(standard_name)})
            fallacies[standard_name].extend(sorted(new_akas))

            used |= overlap
        unused = set(synonyms) - used
        fallacies.update({i: synonyms[i] for i in unused})
        fallacies = {_clean_fallacy_name(k): [_clean_fallacy_name(i) for i in v]
                     for k, v in fallacies.items()}

        big_class2small_classes = {'circular': ['begging the question'],
                                   'cum hoc ergo propter hoc': ['post hoc ergo propter hoc',
                                                                'wrong direction',
                                                                'ignoring a common cause'],
                                   }
        for big_class, small_classes in big_class2small_classes.items():
            for small_class in small_classes:
                fallacies[big_class] += [small_class] + fallacies[small_class]
                del fallacies[small_class]

        aka2std_name = {}
        for standard_name, akas in fallacies.items():
            for i in akas:
                if i:
                    aka2std_name[i] = standard_name
        self.canocalize_fallacy_name = lambda i: aka2std_name.get(i, i)

        fallacy_types = flatten_list([k] + [i for i in v if i] for k, v in fallacies.items())
        cleaned_fallacy_types = {_clean_fallacy_name(i) for i in fallacy_types}

        self.fallacy_types_from_wiki = fallacies

        from efficiency.log import fwrite
        fwrite('\n'.join(sorted(cleaned_fallacy_types)), self.fallacy_list_txt, verbose=True)
        return cleaned_fallacy_types

    def get_fallacy_types(self):
        import os
        if os.path.isfile(self.fallacy_list_csv):
            from efficiency.log import read_csv
            data = read_csv(self.fallacy_list_csv)
            fallacy_types_from_wiki = {i['fallacy_name']: i['alternative_names'].split('; ') for i in data}
            return fallacy_types_from_wiki
        from efficiency.web import get_html_n_save
        html = get_html_n_save(self.fallacy_list_url, self.fallacy_list_html)

        from efficiency.function import flatten_list
        from lxml.html import fromstring, tostring
        from collections import defaultdict
        main_name2akas = defaultdict(list)
        tree = fromstring(html)
        for ul in tree.xpath('//li'):
            justtextstring = tostring(ul)
            ul_text = ''.join(list(ul.itertext()))
            ul_text = ul_text.replace(' - ', ' – ').replace('\xa0– ', ' – ').split('\n')[0]
            if not any(i in ul_text for i in {'fallac', ' – ', "Ignoring a common cause"}):
                continue
            if not ul_text[0].isalpha():
                continue

            fallacy_name = ul_text.split(' – ', 1)[0]
            main_name, *akas = fallacy_name.split(' (', 1)
            main_name = main_name.split('[', 1)[0]  # 'Referential fallacy[42]'
            show_var(['main_name'])
            if main_name == 'Cognitive distortion':
                # stop at the first item in "See Also", which is cognitive distortion
                break
            main_name2akas[main_name] = []
            if akas:
                akas = akas[0]
                from efficiency.function import lstrip_word, rstrip_word
                akas = lstrip_word(lstrip_word(akas, 'also known as '), 'Latin for ')
                akas = akas.rsplit(')', 1)[0]
                akas = akas.replace(', therefore because',
                                    ' therefore because')  # (Latin for "with this, therefore because of this"; correlation implies causation; faulty cause/effect, coincidental correlation, correlation without causation)
                akas = akas.split(', ')
                akas = flatten_list(i.split('; ') for i in akas)
                akas = [lstrip_word(i, 'or ') for i in akas]
                akas = flatten_list(i.split(' or ') for i in akas)
                akas = [i.strip("'").strip('"') for i in akas]
                akas = [i.split('[', 1)[0] for i in akas]

                show_var(['akas'])
                main_name2akas[main_name] = akas
        others = ['red herring']
        for i in others: main_name2akas[i] = []

        import json
        print(json.dumps(main_name2akas, indent=4))
        show_var(['len(main_name2akas)', ])

        from efficiency.log import write_rows_to_csv
        write_rows_to_csv([['fallacy_name', 'alternative_names']] +
                          [[m, '; '.join(a)] for m, a in main_name2akas.items()],
                          self.fallacy_list_csv, verbose=True)
        return main_name2akas


class DataCleaner:
    def __init__(self):
        # raw_data, raw_data_keys = self.load_raw_data(C.raw_data_files)
        # self.keep_only_logical_fallacy_data(raw_data)
        #
        # clean_data, clean_data_keys = self.load_raw_data([C.clean_data_csv])
        # self.normalize_fallacy_types(clean_data)

        normed_data, normed_data_keys = self.load_raw_data([C.clean_data_csv_normed])
        self.save_to_final_form(normed_data)

        # new_data, _ = self.load_raw_data([C.data_merged_csv])

    def get_propaganda_data(self, file):

        lookup = {
            'repetition': 'argument from repetition',
            'flag-waving': 'ad populum',
            'slogans': 'appeal to emotion',
            'doubt': '',
            'thought-terminating cliches': '',
            'exaggeration OR minimisation': '',
        }
        from efficiency.log import read_csv
        propaganda_data = read_csv(file, verbose=True)

        for ix, datum in enumerate(propaganda_data):
            fallacy = datum['logical_fallacy'].lower().replace(',', ' OR ').replace('_',                                                                                                                   ' ')
            fallacy = lookup.get(fallacy, fallacy)
            propaganda_data[ix]['logical_fallacy'] = fallacy

        propaganda_data = [i for i in propaganda_data if i['logical_fallacy']]

        fallacies = [i['logical_fallacy'] for i in propaganda_data]
        from collections import Counter
        Counter(fallacies).most_common()
        # {'article_id': '111111111', 'logical_fallacy': 'appeal to authority', 'text_snippet': 'The next transmission could be more pronounced or stronger', 'sentence': '"The next transmission could be more pronounced or stronger," WHO Director-General Tedros Adhanom Ghebreyesus told reporters in Geneva, insisting that "the issue is serious."'}

        import hashlib

        for ix, row in enumerate(propaganda_data):
            hex_dig = hashlib.shake_256(row['text_snippet'].encode()).hexdigest(5)
            propaganda_data[ix] = {
                'Source URL': row['article_id'] + '-' + hex_dig,
                'Logical Fallacy Types': row['logical_fallacy'],
                'Sentence': row['sentence'],
                'Explanations': '',
                'Rationale': row['text_snippet'],
            }
        return propaganda_data

    def save_to_final_form(self, data):
        '''
        Source URL	"Logical Fallacy Types
                    Key separators: "" & "" to connect multiple logical fallacies,
                    "" > "" to connect hierarchies of logical fallacy types, and
                    "" / "" to connect synonyms of each other"
        Sentence (or small snippets of text)
        Explanations

        source_url	test_name	grade	Logical Fallacy Types	sentence	canocalized_fallacy_types	fallacy_types
        '''
        formatted_data = []
        for row in data:
            new_row = {
                'Source URL': row['source_url'],
                'Logical Fallacy Types': row['canocalized_fallacy_types'],
                'Sentence': row['sentence'],
                'Explanations': '',
                'Rationale': '',
            }
            formatted_data.append(new_row)

        from efficiency.log import write_dict_to_csv
        write_dict_to_csv(formatted_data, C.data_final_format_csv, verbose=True)

        # ---- start merging ----
        from efficiency.log import read_csv, write_dict_to_csv

        data = read_csv(C.data_final_format_csv, verbose=True)
        headers = list(data[0].keys())

        data += read_csv(C.data_manual_csv, verbose=True)
        data += read_csv(C.data_logicallyfallacious_csv, verbose=True)
        # data = self.get_propaganda_data(C.data_propaganda)

        rows = [{header: [v for k, v in row.items() if k.startswith(header)] for header in headers}
                for row in data]
        rows = [{k: v[0] if len(v) else '' for k, v in row.items()} for row in rows]
        rows = self.normalize_fallacy_types(rows, key_sent=headers[2], key_fallacy=headers[1],
                                            key_cano=headers[1], enable_save=False,
                                            enable_save_deleted_content=False, verbose=True)

        write_dict_to_csv(rows, C.data_merged_csv, verbose=True)

        import pandas as pd
        df = pd.read_csv(C.data_merged_csv)
        df.dropna(subset=['Sentence'], inplace=True)
        df = df.rename(columns={"Sentence": "source_article", "Logical Fallacy Types": "logical_fallacies",
                                "Source URL": "original_url", "Explanations": 'explanations'})
        df.to_csv(C.data_merged_csv, index=False)
        return df

    def normalize_fallacy_types(self, clean_data, key_sent='sentence', key_fallacy='Logical Fallacy Types',
                                key_cano='canocalized_fallacy_types', enable_save=True,
                                enable_save_deleted_content=False, verbose=False):
        from collections import defaultdict
        from efficiency.function import flatten_list
        sent2raw_fallacy_types = defaultdict(list)
        for i in clean_data:
            sent2raw_fallacy_types[i[key_sent]].append(i[key_fallacy])
        for line_ix, line in enumerate(clean_data):
            l_f_t = sent2raw_fallacy_types[line[key_sent]]
            fallacy_types = set(flatten_list([C.contain_which_fallacy(i) for i in l_f_t]))

            # if not fallacy_types: import pdb;pdb.set_trace()

            clean_data[line_ix][key_cano] = '; '.join(
                sorted({C.canocalize_fallacy_name(i) for i in fallacy_types}))
            if enable_save: clean_data[line_ix]['fallacy_types'] = '; '.join(sorted(fallacy_types))
            if not clean_data[line_ix][key_cano]:
                if verbose:
                    show_var(['line_ix', 'l_f_t'], joiner=', ')

        show_var(['len(clean_data)'])
        uniq2data = {i[key_sent] + '\t' + i[key_cano]: i for i in clean_data}
        # uniq2data = {i['sentence']: i for i in clean_data}
        clean_data = list(uniq2data.values())
        show_var(['len(clean_data)'])

        all_types = [i[key_cano] for i in clean_data]

        clean_data = sorted(clean_data, key=lambda i: all_types.count(i[key_cano]), reverse=True)
        if enable_save_deleted_content:
            not_aligned_data = [i for i in clean_data if not i[key_cano]]
            all_raw_logics = [i[key_fallacy].lower() for i in not_aligned_data]
            not_aligned_data = sorted(not_aligned_data, key=lambda i: (
                - all_raw_logics.count(i[key_fallacy].lower()), i[key_fallacy]))

        clean_data = [i for i in clean_data if i[key_cano]]
        if enable_save:
            from efficiency.log import write_dict_to_csv
            write_dict_to_csv(clean_data, C.clean_data_csv_normed, verbose=True)
        if enable_save_deleted_content:
            write_dict_to_csv(not_aligned_data, C.data_csv_unaligned, verbose=True)
        show_var(['len(clean_data)'])
        print('finished normalization')
        return clean_data

    def keep_only_logical_fallacy_data(self, raw_data):
        # ['test_name', 'Logical Fallacy Types', 'sentence', 'grade', 'source_url', ]
        from efficiency.function import flatten_list, lstrip_word, rstrip_word
        from nltk.tokenize import word_tokenize

        bad_ixs = []
        conc_ixs = []
        for ix, line in enumerate(raw_data):
            sent = line['sentence']
            # if 'Attacks the arguer instead of the argument' in sent:
            #     import pdb;pdb.set_trace()
            toked = ' '.join(word_tokenize(sent)).lower()
            if any(i.lower() in toked for i in
                   {'pictured', 'this ad ', 'the advertisement', 'this advertisement', 'this image', 'following image', 'the picture',
                    'the above picture', 'used in the photo', 'this cartoon',
                    'this picture', 'match the definition with the fallacy', '_'}):
                bad_ixs.append(ix)
            for b_q_p in C.bad_ques_pref + C.bad_defi_only:
                sent = lstrip_word(sent, b_q_p).strip()
            for u_q_s in C.bad_ques_suff:
                sent = sent.rsplit(u_q_s, 1)[0].strip()

            raw_data[ix]['sentence'] = sent
        for ix, line in enumerate(raw_data):
            sent = line['sentence']
            toked = ' '.join(word_tokenize(sent)).lower()
            if any((i.lower() in toked) or (i.lower() in sent.lower()) for i in C.bad_incl_terms):
                conc_ixs.append(ix)

        show_var(['len(raw_data)'])
        # bad_sents = [raw_data[ix]['sentence'] for ix in conc_ixs]
        # import pdb;pdb.set_trace()
        # print('\n'.join(bad_sents))
        non_image_data = [line for ix, line in enumerate(raw_data) if ix not in (set(bad_ixs) | set(conc_ixs))]
        show_var(['len(raw_data)'])
        uniq2raw_data = {i['sentence'] + '\t' + i['Logical Fallacy Types']: i for i in non_image_data}
        non_image_data = list(uniq2raw_data.values())
        show_var(['len(raw_data)'])

        answers = [i['Logical Fallacy Types'] for i in non_image_data]
        from collections import Counter
        freq_answers = {k for k, v in Counter(answers).items() if v > 10}
        # fallacy_types = freq_answers | fallacy_types

        terms = {'fallac', 'logic', 'thinking', 'reason', 'debate', 'propaganda', 'persua'}

        criteria = lambda i: any(t in i[k].lower() for t in terms for k in ['test_name', 'sentence']) \
                             or C.contain_which_fallacy(i['Logical Fallacy Types'])
        criteria_non_empty = lambda i: bool(i['Logical Fallacy Types']) and (
                len(i['sentence'].split()) >= C.min_sent_len)
        criteria_must = lambda i: any(
            C.contain_which_fallacy(i[k]) for k in ['Logical Fallacy Types', 'sentence'][:1])

        non_empty_data = list(filter(criteria_non_empty, non_image_data))
        show_var(['len(non_empty_data)'])
        # clean_data = list(filter(criteria_must, clean_data))
        # show_var(['len(clean_data)'])
        clean_data = list(filter(criteria, non_empty_data))
        show_var(['len(clean_data)'])

        from efficiency.log import write_dict_to_csv
        write_dict_to_csv(clean_data, C.clean_data_csv, verbose=True)
        deleted = [i for i in raw_data if i not in clean_data]
        write_dict_to_csv(deleted, C.deleted_data_csv, verbose=True)
        return clean_data

    def load_raw_data(self, files):
        from efficiency.log import read_csv
        data = []
        for file in files:
            data += read_csv(file, verbose=True)

        key_list = [i.keys() for i in data]  # 'test_name', 'grade', 'source_url', 'Logical Fallacy Types', 'sentence'
        from efficiency.function import flatten_list
        all_keys = set(flatten_list(key_list))
        all_keys = [i for i in all_keys if i]
        return data, all_keys


def main():
    dc = DataCleaner()


if __name__ == '__main__':
    C = Constants()
    main()
