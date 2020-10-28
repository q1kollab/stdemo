import streamlit as st
import re
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelWithLMHead
import spacy 
import en_core_web_sm
import fitz
from operator import itemgetter
import copy
import base64

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def bimg2utf(bimg):
    return base64.b64encode(bimg).decode('utf8')

MAX_LENGTH = int(10000)
def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

@st.cache
def generate_questions(answers, context, max_length=64):
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    model.to(device)
    
    qa_pairs = []
    for answer in answers:
        input_text = "answer: %s  context: %s </s>" % (answer, context)

        features = tokenizer([input_text], return_tensors='pt')
        output = model.generate(input_ids=features['input_ids'].to(device), 
                                attention_mask=features['attention_mask'].to(device),
                                max_length=max_length)
        question = tokenizer.decode(output[0])[10:]
        qa_pairs.append((question, answer))

    return qa_pairs

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

@st.cache
def generate_keywords(text):
    
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    words = [str(x) for x in doc]
    labels = [str(x.ent_type_) for x in doc]
    ents, names = [], []
    for i, (word, label) in enumerate(zip(words, labels)):
        if label == '':
            continue
        elif labels[i]==labels[i-1]:
            ents[-1]+=' '+word
        else:
            ents.append(word)
            names.append(label)
    return unique(ents)



def _fonts(doc, granularity=True):
    """Extracts fonts and their usage in PDF documents.
    :param doc: PDF document to iterate through
    :type doc: <class 'fitz.fitz.Document'>
    :param granularity: also use 'font', 'flags' and 'color' to discriminate text
    :type granularity: bool
    :rtype: [(font_size, count), (font_size, count}], dict
    :return: most used fonts sorted by count, font style information
    """
    styles = {}
    font_counts = {}

    for page in doc:
        blocks = page.getText("dict")["blocks"]
        for b in blocks:  # iterate through the text blocks
            if b['type'] == 0:  # block contains text
                for l in b["lines"]:  # iterate through the text lines
                    for s in l["spans"]:  # iterate through the text spans
                        if granularity:
                            identifier = "{0}_{1}_{2}_{3}".format(s['size'], s['flags'], s['font'], s['color'])
                            styles[identifier] = {'size': s['size'], 'flags': s['flags'], 'font': s['font'],
                                                  'color': s['color']}
                        else:
                            identifier = "{0}".format(s['size'])
                            styles[identifier] = {'size': s['size'], 'font': s['font']}

                        font_counts[identifier] = font_counts.get(identifier, 0) + 1  # count the fonts usage

    font_counts = sorted(font_counts.items(), key=itemgetter(1), reverse=True)

    if len(font_counts) < 1:
        raise ValueError("Zero discriminating fonts found!")

    return font_counts, styles

def _font_tags(font_counts, styles):
    """Returns dictionary with font sizes as keys and tags as value.
    :param font_counts: (font_size, count) for all fonts occuring in document
    :type font_counts: list
    :param styles: all styles found in the document
    :type styles: dict
    :rtype: dict
    :return: all element tags based on font-sizes
    """
    p_style = styles[font_counts[0][0]]  # get style for most used font by count (paragraph)
    p_size = p_style['size']  # get the paragraph's size

    # sorting the font sizes high to low, so that we can append the right integer to each tag 
    font_sizes = []
    for (font_size, count) in font_counts:
        if re.search('\_', font_size):
            font_size = font_size.split('_')[0]
        font_sizes.append(float(font_size))
    font_sizes.sort(reverse=True)

    # aggregating the tags for each font size
    idx = 0
    size_tag = {}
    for size in font_sizes:
        idx += 1
        if size == p_size:
            idx = 0
            size_tag[size] = '<p>'
        if size > p_size:
            size_tag[size] = '<h{0}>'.format(idx)
        elif size < p_size:
            size_tag[size] = '<s{0}>'.format(idx)

    return size_tag

def _flags_read(flags):
    """Make font flags human readable."""
    l = []

    if flags & 2 ** 4:
        l.append("bold")

    if flags & 2 ** 1:
        l.append("italic")

    if flags & 2 ** 0:
        l.append("superscript")

#     if flags & 2 ** 2:
#         l.append("serifed")
#     else:
#         l.append("sans")
#     if flags & 2 ** 3:
#         l.append("monospaced")
#     else:
#         l.append("proportional")
    return l

class PDF:
    def __init__(self, granularity=True):
        self.granularity = granularity
        self.font_counts = None
        self.font_styles = None
        self.font_tags = None
    

    
    def __call__(self, path):
        """Scrapes headers & paragraphs from PDF and return texts with element tags.
        :param doc: PDF document to iterate through
        :type doc: <class 'fitz.fitz.Document'>
        :param self.font_tags: textual element tags for each size
        :type self.font_tags: dict
        :rtype: list
        :return: texts with pre-prended element tags
        """
        doc = fitz.open(path)
        self.font_counts, self.font_styles = _fonts(doc)
        self.font_tags = _font_tags(self.font_counts, self.font_styles)
        
        pages = {}  # list with headers and paragraphs
        first = True  # boolean operator for first header
        previous_s = {}  # previous span
        prev_bold = False

        for j, page in enumerate(doc):
            blocks = page.getText("dict")["blocks"]
            
            for image_index, img in enumerate(page.getImageList(), start=1):
                # get the XREF of the image
                xref = img[0]
                # extract the image bytes
                base_image = doc.extractImage(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                if j+1 in pages:
                    pages[j+1].append('{}{}'.format('<im>', bimg2utf(image_bytes)))
                else:
                    pages[j+1] = ['{}{}'.format('<im>', bimg2utf(image_bytes))]
                
            for b in blocks:  # iterate through the text blocks
                if b['type'] == 0:  # this block contains text
                    block_string = ""  # text found in block
                    for l in b["lines"]:  # iterate through the text lines
                        span_length = len(l["spans"])
                        
                        for i, s in enumerate(l["spans"]):  # iterate through the text spans
                            final_span = i==span_length-1
                            
                            if s['text'].strip():  # removing whitespaces:
                                text = s['text'].strip()
                                text_size = s['size']
                                text_bold = 'bold' in _flags_read(s['flags'])
                                bold_end = False
                                
                                if first:
                                    if text_bold and not prev_bold:
                                        text = '<b>'+text
                                        if final_span:
                                            text = text+'</b>'
                                            bold_end = True
                                            
                                    first = False
                                    block_string = self.font_tags[text_size] + text
                                    prev_size = text_size
                                    prev_bold = text_bold if not bold_end else False
                                else:
                                    if text_size == prev_size:
                                        
                                        if block_string == "" or block_string and all((c == "|") for c in block_string):
                                            # new block or block_string only contains pipes, so append size tag
                                            if text_bold and not prev_bold:
                                                text = '<b>'+text
                                                if final_span:
                                                    text = text+'</b>'
                                                    bold_end = True
                                                    
                                            block_string = self.font_tags[text_size] + text

                                        else:  # in the same block, so concatenate strings
                                            if text_bold and not prev_bold:
                                                text = '<b>'+text
                                                if final_span:
                                                    text = text+'</b>'
                                                    bold_end = True
                                            elif prev_bold:
                                                block_string +='</b>'
                                                bold_end = True
                                                    
                                            block_string += " " + text
                                    else:
                                        if text_bold and not prev_bold:
                                            text = '<b>'+text
                                            if final_span:
                                                text = text+'</b>'
                                                bold_end = True
                                            
                                        if len(block_string)>0 and re.search('[a-zA-Z0-9]', block_string):
                                            if j+1 in pages:
                                                pages[j+1].append(re.sub(r'\s+',' ', block_string))
                                            else:
                                                pages[j+1] = [re.sub(r'\s+',' ', block_string)]
                                        block_string = self.font_tags[text_size] + text
                                    prev_size = text_size
                                    prev_bold = text_bold if not bold_end else False
                        # new block started, indicating with a pipe
                        block_string += "|"

                    if len(block_string)>0 and re.search('[a-zA-Z0-9]', block_string):
                        if j+1 in pages:
                            pages[j+1].append(re.sub(r'\s+',' ', block_string))
                        else:
                            pages[j+1] = [re.sub(r'\s+',' ', block_string)]

            image_list = page.getImageList()
            
        pages_split = {}
        
        for k,content in pages.items():
            pages_split[k] = []
            for line in content:
                if re.search(r'\.{4}', line):
                    pages_split[k] += [line.strip()]
                else:
                    tag = re.search('\<.{1,3}\>', line).group()
                    for split in re.split('\|{2,10}', line):
                        if len(split)>0:
                            if re.search(r'(?<=\>).+', split) is None:
                                pages_split[k] += ['{}{}'.format(tag, split.strip())]
                            else:
                                pages_split[k] += [split.strip()]
                        
        return pages_split
    


class TMparser:
    def __init__(self, pages, images=False):
        self.pages = pages
        self.content, self.page_ref = self.aggregate_content(images)
        self.toc = self.get_toc()
        self.sections_clean, self.sections_dirty, self.sections_discard = self.make_sections()
        self.sections_final = self.make_section_nums()
        self.sections_final_cxt = self.make_context(self.sections_final)
        
    def get_page(self, num):
        assert num in self.page_ref
        start, end = self.page_ref[num]
        return self.content[start:end]
    
    def get_section(self, num, stype='clean'):
        if stype=='clean':
            title = list(self.sections_clean.keys())[num]
            return title, self.sections_clean[title]
        elif stype=='dirty':
            title = list(self.sections_dirty.keys())[num]
            return title, self.sections_dirty[title]
        elif stype=='discard':
            title = list(self.sections_discard.keys())[num]
            return title, self.sections_discard[title]
        
    def aggregate_content(self, images):
        content = []
        page_ref = {}
        
        for k,v in self.pages.items():
            start = len(content)
            if images:
                content+=v
            else:
                content+=[x for x in v if not x.startswith('<im>')]
            end = len(content)
            page_ref[k] = (start, end)
        return content, page_ref
    
    def get_toc(self):
        toc_raw = []
        num = ''
        
        for i, line in enumerate(self.content):
            prev_line = self.content[i-1] if i>0 else ''
            next_line = self.content[i+1] if i<i<len(self.content)-1 else ''
            
            line = re.sub(r'\<b\>|\<\/b\>', '', re.sub('\|+', '|', re.search(r'(?<=\>).+', line).group()))
            
            if re.search(r'^[a-zA-Z]+\s[0-9]+[^\.\,0-9]+$|^[0-9]+\s[a-zA-Z]+[^\.\,0-9]+$', line) and (
                re.search(r'\.{4}', prev_line) or 
                re.search(r'\.{4}', next_line)):
                num = re.search(r'[0-9]+', line).group()
                print(line, num)
                
            if re.search(r'\.{4}', line) is not None:
                if re.search('^[0-9]', line) is None:
                    line = '{} {}'.format(num, line)
                if len(re.findall(r'\.{4}[a-zA-Z0-9\s\-\–\(\)]', line))>1:
                    toc_raw += re.split(r'(?<=[0-9]\|)\s', line)
                else:
                    toc_raw.append(line)

        toc = []
        for line in toc_raw:
            renum = re.search(r'^[0-9\.\-\–]*(?=\s)', line)
            num = '' if renum is None else renum.group().strip()
            name = re.search(r'(?<=^{}).+?(?=\.\.\.)'.format(re.escape(num)), line).group().strip()
            page = re.search(r'(?<=\.\.\.)[^\.]*[a-zA-Z0-9]+[0-9\.\-\–\s]{0,5}(?=\|$)', line).group().strip()
            
            toc.append((num, name, page))
        return toc
    
    def _sections(self):
        sections = []
        start = 0

        for section, name, page in self.toc:
            name_tag = re.escape(re.search('.+(?=\:)|.+(?=\|)|.+$', name).group())

            found_name = False
            for i, line in enumerate(self.content[start:]):
                if re.search(r'\.{4}', line):
                    continue
                
                text_tags = re.findall('(?<=\<b\>).+?(?=\<\/b\>)', line)
                text_tags += [re.sub(r'\<b\>|\<\/b\>', '', re.search(r'(?<=\>).+', line).group())]
                
                for text_tag in text_tags:
                    if re.search('^[a-zA-Z\(\)\-\–\.]{1,4}\s|^[0-9\(\)\-\–\.]{2,6}\s', text_tag):
                        text_tag = re.search('(?<=\s).*', text_tag).group()

                    if re.search(r'^{}'.format(name_tag), text_tag):
                        found_name = True
                        sections.append((i+start, name, line))
                        start+=i
                        break
                if found_name:
                    break

            if not found_name and page.isdigit():
                idx1, idx2 = self.page_ref[int(page)]
                name_tag = re.escape(re.search('.+?(?=\s)|.+$', name).group().lower())
                for j, line in enumerate(self.content[idx1:idx2]):
                    text_tag = re.sub(r'\<b\>|\<\/b\>', '', re.search(r'(?<=\>).+', line).group()).lower()
                    if re.search(r'{}'.format(name_tag), text_tag):
                        found_name = True
                        sections.append((j+idx1, name, line))
                        break
            if not found_name:   
                sections.append((-1, name, -1))
        return sections
    
    def _clean_section(self, section):
        clean, dirty = [], []
        for sec in section:
            sec = re.sub(r'\uf0a7', '', re.search(r'(?<=\>).+', sec).group().replace('|', '')).strip()
            numbers = sum(c.isdigit() for c in sec)
            letters = sum(c.isalpha() for c in sec)
            
            if (re.search('^\<b\>.+\<\/b\>$', sec) or re.search(r'\.{4}', sec) 
                or letters<=25 or (numbers+10e-5)/(letters+10e-5)>0.3) and not re.search(r'^[a-z]', sec):
                dirty.append(sec)
            else:
                clean.append(re.sub(r'\<b\>|\<\/b\>', '', sec))
        return clean, dirty
    
    def _super_clean(self, section):
        clean = []
        for sec in section:
            subsec = re.split(r'\s(?=\•)', sec) if re.search(r'\s\•', sec) else [sec]
            for sub in subsec:
                if re.search(r'^[a-z]', sub) and len(clean)>0:
                    clean[-1]+=' {}'.format(sub)
                else:
                    clean.append(sub)
        return clean
                
    
    def make_sections(self):
        sections_clean, sections_dirty, section_discard = {}, {}, {}
        
        _sections = [x for x in self._sections() if x[0]>0]
        for i, (idx,name,_) in enumerate(_sections):
            
            idx_next = _sections[i+1][0] if i<len(_sections)-1 else len(self.content)-1
            
            clean, dirty = self._clean_section(self.content[idx:idx_next])
            
            if len(''.join(clean))>100:
                sections_clean[name] = self._super_clean(clean)
                sections_dirty[name] = dirty
            else:
                section_discard[name] = clean+dirty
            
            if idx_next<idx:
                print(name)
                break
        return sections_clean, sections_dirty, section_discard
    
    def make_section_nums(self):
        section_nums = {}
        for x in self.toc:
            if x[1] in section_nums:
                pass
            else:
                section_nums[x[1]] = x[0]
        
        section_nums_clean = {}
        
        for k in self.sections_clean:
            section_nums_clean[k] = section_nums[k].replace('-', '.').strip('.')
            
        all_nums = list(section_nums_clean.values())
        existing_nums = [x for x in all_nums if len(x)>0]
        
        if len(existing_nums)==0:
            for k,v in zip(list(section_nums_clean.keys()), list(range(len(all_nums)))):
                section_nums_clean[k] = v
        elif len(all_nums[0])==0:
            fill_num = int(existing_nums[0][0])-1
            sub_num = 0
            for k,v in section_nums_clean.items():
                if len(v)>0:
                    break
                else:
                    section_nums_clean[k]='{}.{}'.format(fill_num, sub_num)
                    sub_num+=1
        elif all_nums[0]==all_nums[1]:
            prev_num = all_nums[0]
            sub_num = 0
            
            for k,v in section_nums_clean.items():
                if v==prev_num:
                    section_nums_clean[k]='{}.{}'.format(v, sub_num)
                    sub_num+=1
                else:
                    sub_num=0
                    section_nums_clean[k]='{}.{}'.format(v, sub_num)
        
        sections_clean_numbered = {}
        for k,v in self.sections_clean.items():
            sections_clean_numbered['{} {}'.format(section_nums_clean[k], k)] = v
        return sections_clean_numbered
    
    def make_context(self, sections):
        sections_cxt = {}
        sections_copy = copy.deepcopy(sections)
        for k,v in sections_copy.items():
            cxt = ''
            for line in v:
                if re.search(r'\s[a-z]+\s', line):
                    if re.search('^[0-9\(\)\-\–\.]{1,6}\s', line):
                        line = re.search('(?<=\s).*', line).group()
                    cxt += ' {}'.format(re.sub(r'[^a-zA-Z0-9\s\(\)\*\'\"\.\,\;\?\!\-\/]+', '', line))
            sections_cxt[k] = re.sub(r'\(.{1,17}\)', '', cxt)
        return sections_cxt

def pdf_analyzer(path):
    pdfparser = PDF()
    pdfdict = pdfparser(path)
    return TMparser(pdfdict)
    
def wrap_by_word(s, n):
    '''returns a string where \\n is inserted between every n words'''
    a = s.split()
    ret = ''
    for i in range(0, len(a), n):
        ret += ' '.join(a[i:i+n]) + '\n '

    return ret

@st.cache
def pretty_print(content, n):
    for i in range(len(content)):
        content[i] = wrap_by_word(content[i], n)
    return content
        
    
import glob
    
def run():
    data_paths = glob.glob('data/DoD/*')
    data_names = [re.search('(?<=\\\).+(?=\.[pdfPDF]{3}$)', x).group() for x in data_paths]
    text_sources = dict(zip(data_names, data_paths))
    text_parser = {}
    text_final = {}
    text_cxt = {}
    
    name = st.sidebar.selectbox("Choose technical manual", list(text_sources.keys()), 0)
    text_parser[name] = pdf_analyzer(text_sources[name])
    text_final[name] = text_parser[name].sections_final
    text_cxt[name] = text_parser[name].sections_final_cxt
    
    sec = st.sidebar.selectbox("Choose section", list(text_final[name].keys()), 0)
    content = text_final[name][sec]
    context = text_cxt[name][sec]
    
    st.header('Keyword-Based Question Generator')
    st.markdown('***First our entity recognition algorithm searches for key words in the content below. Then our question generator creates questions for which the key words are the answers. Click the button below to give it a try!***')
    
    num_q = st.slider('Number of questions to generate', 0, 20, 8)
    if st.button("Generate Questions"):
        ents = generate_keywords(context)[:num_q]
        if len(ents)>0:
            qa_pairs = generate_questions(ents, context)
            for q, a in qa_pairs:
                st.text('Question: {}\nAnswer: {}'.format(wrap_by_word(q, 10).strip('\n '), a))
        else:
            st.markdown('***Unfortunately no key words were found in this section. Try running this on a section with more substantive content, or try entering your own key words below!***')

    st.header('User-Assisted Question Generator')
    st.markdown('***If you enter your own answers in the text box below, our question generator will create a corresponding question.***')
    
    user_answer = st.text_area("Enter answer", "")
    if st.button("Get Question"):
        qa_pairs_user = generate_questions([user_answer], context)
        q_user, a_user = qa_pairs_user[0]
        st.text('Question: {}\nAnswer: {}'.format(wrap_by_word(q_user, 10).strip('\n '), a_user))
    
    st.header(name)
    st.subheader(sec)
    
    st.text('\n'.join(pretty_print(content, 10)))

if __name__ == "__main__":
    run()