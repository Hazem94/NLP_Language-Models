from collections import Counter
import random, operator, os, sys, time, math



def read_text_file():
    """
    A function to read the input data from the file
    :return: 1) train_text: the training data text
             2) test_text: the test data text
    """
    # Open the input file and handle I/O exceptions
    train_text = []
    test_text = []
    text = []

    num_lines = sum(1 for line in open(sys.argv[1]))  # Count the number of lines
    num_train_lines = int(0.6 * num_lines)  # Number of lines of the training text
    num_test_lines = num_lines - num_train_lines
    i = 0

    # Read %60 of the text as training data set and the remaining %40 as test set
    try:
        file = open(sys.argv[1], 'r')
        text = [str("<s> ") + line.rstrip('\n') + str(" </s> ") for line in file]
        file.close()
    except IOError:
        print("I/O error occurred, check your input file!\nExiting the program ... ")
        exit(-1)
    while i < num_lines:
        if i <= num_train_lines:
            train_text.append(text[i])
        else:
            test_text.append(text[i])
        i += 1
    return ''.join(train_text), ''.join(test_text)



def delete_out_file():
    # Delete result.txt file if already exists
    os.remove(sys.argv[2]) if os.path.exists(sys.argv[2]) else None



def out_on_file(text):
    """
    A function used to print on a text file named "result.txt"
    :param text: The string to be printed out on "result.txt"
    :return: Nothing
    """
    # Open the output file and handle I/O exceptions
    try:
        file = open(sys.argv[2], 'a')
        file.writelines(text)
        file.close()
    except IOError as err:
        print("I/O error occurred with your output file! [Error: {}] \nExiting the program ... ".format(err))
        exit(-1)



def build_unigram_freq_model(train_tokens):
    """
    Function used to build unigram frequency model from training data set
    :param train_tokens: a dictionary, the tokens used in building the model
    :return: 1) unigram: a dictionary, the unigram frequency model
             2) v: an integer, the number of types 'Vocabulary size' of unigram frequency model
             3) n: an integer, the number of tokens of unigram frequency model
    """

    unigram = {}
    tokens = [item for item in train_tokens if item != "<s>"]

    # Tokens count
    n = len(tokens)

    # Unigram is a dictionary that maps each word with its occurrence number, like: 'home': 2 ....
    unigram = dict(Counter(tokens))

    # vocabulary 'types' count
    v = len(unigram.keys())

    return unigram, v, n



def count_freq_bigram(first, second, tokens):
    """
    A function used to count the frequencies of having first followed by second words in the text or tokens
    :param  first: a string, the first word
    :param second: a string, the second word following the first word
    :param tokens: a list, the tokens of the train text
    :return: the count C( second | first )
    """
    counter = 0
    for i in range(len(tokens) - 1):
        if first == tokens[i] and second == tokens[i + 1]:
            counter += 1
    return counter



def count_freq_trigram(first, second, third, tokens):
    """
    A function used to count the frequencies of having 'first second third' following each other in the text
    :param first: a string, the first word
    :param second: a string, the word following the first word
    :param third: a string, the word following the second word
    :param tokens: a string, the tokens of the train text
    :return: counter: an integer, the count C( third | first second )
    """
    counter = 0
    for i in range(len(tokens) - 2):
        if first == tokens[i] and second == tokens[i + 1] and third == tokens[i + 2]:
            counter += 1
    return counter



def build_bigram_freq_model(train_tokens):
    """
    Function used to build bigram frequency model from training data set
    :param   train_tokens: a list, the tokens used in building up the model
    :return: 1) bigram: a dictionary, the bigram frequency model "frequency of each two sequence words"
             2) bigram_freq: a dictionary, tokens frequency dictionary
             3) v: an integer, the number of types 'Vocabulary size' of bigram frequency model
             4) n: an integer, the number of tokens of bigram frequency model
    """

    # Bigram model that is the occurrences of each two sequential tokens
    # For example: 'I am a student', C('am'| 'I') of the form 'I am' means: 'I' followed by 'am'
    bigram = {}

    tokens = [item for item in train_tokens]

    # Tokens count
    n = len(tokens)

    # freq_bigram is a dictionary that maps each word with its frequency number, like: 'home': 2 ....
    freq_bigram = dict(Counter(tokens))

    # Vocabulary 'types' count
    v = len(freq_bigram.keys())

    # Setting up 'bigram' the frequency dictionary that contains the frequency of each two sequence words
    for i in range(len(tokens) - 1):
        word = tokens[i] + ' ' + tokens[i + 1]
        bigram[word] = count_freq_bigram(word.split()[0], word.split()[1], tokens)

    return bigram, freq_bigram, v, n



def build_trigram_freq_model(train_tokens):
    """
    A function used to build 'trigram' model 'the frequencies of series of strings'
    :param train_tokens: a list, the tokens of train data set
    :return: 1) trigram: a dictionary, the trigram frequency model
             2) freq_trigram: a dictionary, the frequencies of each three sequence tokens
             3) v: an integer, the number of types 'Vocabulary size' of unigram frequency model
             4) n: an integer, the number of tokens of unigram frequency model
             5) (start_end_punct_count/2): an integer, the count of <s> or </s> that are added to the train text
    """

    # trigram model that contains the occurrences of each three sequential tokens
    # For example: '<s> <s> I am a student </s> </s>', C('I am'| 'a') of the form '
    # I|am|a' means: 'I' followed by 'am' followed by 'a'
    trigram = {}

    # counter to count # of <s> or </s> 'punctuation refer to end or start of a sentence' to be added
    start_end_punct_count = 0

    tokens = []
    # Duplicating <s> and </s> in our tokens to perform trigram model
    for token in train_tokens:
        if token in ('<s>', '</s>'):
            tokens.extend([token, token])
            start_end_punct_count += 1
        else:
            tokens.append(token)

    # Tokens count
    n = len(tokens)

    # init_trigram is a dictionary that maps each token with its occurrence number, like: 'home': 2 ....
    freq_trigram = dict(Counter(tokens))

    # Vocabulary 'types' count
    v = len(freq_trigram.keys())

    # Setting up the frequency dictionary and finding out the frequencies of each three sequence words
    for i in range(len(tokens) - 2):
        word = tokens[i] + ' ' + tokens[i + 1] + ' ' + tokens[i + 2]
        trigram[word] = count_freq_trigram(word.split()[0], word.split()[1], word.split()[2], tokens)

    return trigram, freq_trigram, v, n, (start_end_punct_count / 2)



def build_pr_unigram(unigram_freq_model, unigram_v, unigram_n):
    """
    A function used to build the unigram probability model
    :param unigram_freq_model: a dictionary, the unigram frequency model
    :param unigram_v: an integer, the number of types 'Vocabulary size' of unigram frequency model
    :param unigram_n: an integer, the number of tokens of unigram frequency model
    :return: 1) unsmoothed: a dictionary, the unsmoothed unigram model
             2) smoothed: a dictionary, the smoothed unigram model
    """

    smoothed = {}
    unsmoothed = {}
    # For each word in frequency unigram divide the frequency by the number of tokens for unsmoothed, and for smoothed
    # use Laplace smoothing "add one to each frequency then normalize by dividing by v, vocabulary size
    for word in unigram_freq_model.keys():
        unsmoothed[word] = unigram_freq_model[word] / unigram_n
        smoothed[word] = (unigram_freq_model[word] + 1) / (unigram_n + unigram_v)
    return unsmoothed, smoothed



def build_pr_bigram(bigram_freq_model, tokens_freq_bigram, unigram_v):
    """
    A function used to build the probability model from the frequency model
    :param bigram_freq_model: a dictionary, the bigram frequency model
    :param tokens_freq_bigram: a dictionary, the bigram tokens frequency model
    :param unigram_v: an integer, the number of types 'Vocabulary size' of unigram frequency model
    :return:1) unsmoothed: a dictionary, the unsmoothed bigram probability model
            2) smoothed: a dictionary, the smoothed bigram probability model
    """
    smoothed = {}
    unsmoothed = {}
    for clause in bigram_freq_model.keys():  # keys of the form "I am" 'I' followed by 'am'
        unsmoothed[clause] = bigram_freq_model[clause] / tokens_freq_bigram[clause.split()[0]]
        smoothed[clause] = (bigram_freq_model[clause] + 1) / (tokens_freq_bigram[clause.split()[0]] + unigram_v)
    return unsmoothed, smoothed



def build_pr_trigram(trigram_freq_model, bigram_freq_model, bigram_v, start_end_punct_count):
    """
    A function used to build the probability model from the frequency model
    :param trigram_freq_model: a dictionary, the trigram frequency model
    :param bigram_freq_model: a dictionary, the bigram tokens frequency model
    :param bigram_v: an integer, the number of types 'Vocabulary size' of bigram frequency model
    :param start_end_punct_count: an integer, the count of <s> or </s>
    :return:1) unsmoothed: a dictionary, the unsmoothed trigram probability model
            2) smoothed: a dictionary, the smoothed trigram probability model
    """

    smoothed = {}
    unsmoothed = {}
    # Add the count of  "<s> <s>" and "</s> </s>" which is the same as start_end_punct_count
    bigram_freq_model["<s> <s>"] = start_end_punct_count
    bigram_freq_model["</s> </s>"] = start_end_punct_count

    for clause in trigram_freq_model.keys():
        denom = bigram_freq_model[clause.split()[0] + ' ' + clause.split()[1]]  # denominator
        unsmoothed[clause] = trigram_freq_model[clause] / denom
        smoothed[clause] = (trigram_freq_model[clause] + 1) / (denom + bigram_v)

    return unsmoothed, smoothed



def build_pr_distr_unigram(unigram):
    """
    A function used to build the probability distribution of unigram model both smootehd and unsmoothed
    :param   unigram: a dictionary, the unigram model
    :return: pr_distr: a list, the probability distribution of unigram
    """

    # Building up the probability distribution of the form [[string, probability] ... ]
    pr_distr = list([k, unigram.get(k)] for k in unigram.keys())

    # Sorting the probability among the distribution table
    pr_distr.sort(key=operator.itemgetter(1))

    # Building the cumulative probabilities by summing up the previous pr.s
    for i in range(1, len(unigram)):
        new_value = pr_distr[i][1] + pr_distr[i - 1][1]
        pr_distr[i][1] = new_value

    return pr_distr



def find_word_str_matches_pr(pr_distr, w_pr):
    """
    A function used to find and pick up a word that its probability matches the passed number 'w_pr' using the
    probability distribution
    :param pr_distr: a list, the probability distribution of N-Gram
    :param w_pr: a float, the random generated probability to match a word from the distribution
    :return: 1) the matched string or word
             2) the probability of the matched word
    """

    for j in range(len(pr_distr) - 1):
        if w_pr >= pr_distr[j][1] and w_pr < pr_distr[j + 1][1]:
            return pr_distr[j][0], pr_distr[j][1]
    return pr_distr[len(pr_distr) - 1][0], pr_distr[len(pr_distr) - 1][1]



def generate_unigram_emails(pr_distr_unsmoothed_unigram, pr_distr_smoothed_unigram):
    """
    A function used to generate emails from both smoothed and unsmoothed unigram
    :param pr_distr_unsmoothed_unigram: a list, the unsmoothed unigram probability distribution
    :param pr_distr_smoothed_unigram: a list, the smoothed unigram probability distribution
    :return: emails: a string, the emails generated using both smootehd and unsmoothed unigram models
    """

    emails = "Emails generated using unsmoothed unigram:\n\n"
    end_sent_punct = ('!', '?', '.')  # End of sentence punctuations

    # Generating 10 emails unsmoothed
    for email_count in range(10):
        emails += '%2d) ' % (email_count + 1)
        # Generate a sentence
        sentence_pr_log = 0.0  # sentence probability
        words_count = random.randint(1, 30)  # words_count: A random number that does not exceed 30
        for i in range(words_count):
            w_pr = random.uniform(0, 1)  # Randomly generated pr of the next word
            w, w_pr = find_word_str_matches_pr(pr_distr_unsmoothed_unigram, w_pr)  # The predicted next word
            # check if w is 'end of sentence punctuation' </s> then, stop
            sentence_pr_log += math.log2(w_pr)
            if w == "</s>":
                break
            elif w in end_sent_punct:
                emails += ' ' + w + ' '
                break
            emails += ' ' + w + ' '
        emails += '[' + str(sentence_pr_log) + ']\n'

    emails += "\n\n" + 40 * "=" + '\n\n'

    emails += "Emails generated using smoothed unigram:\n\n"

    # Generating 10 emails unsmoothed
    for email_count in range(10):
        emails += '%2d) ' % (email_count + 1)
        # Generate a sentence
        sentence_pr_log = 0.0   # sentence probability in log space
        words_count = random.randint(1, 30)  # words_count: A random number that does not exceed 30
        for i in range(words_count):
            w_pr = random.uniform(0, 1)  # Randomly generated pr of the next word
            w, w_pr = find_word_str_matches_pr(pr_distr_smoothed_unigram, w_pr)  # The predicted next word
            # check if w is 'end of sentence punctuation' </s> then, stop
            sentence_pr_log += math.log2(w_pr)
            if w == "</s>":
                break
            elif w in end_sent_punct:
                emails += ' ' + w + ' '
                break
            emails += ' ' + w + ' '
        emails += '[' + str(sentence_pr_log) + ']\n'

    emails += "\n\n" + 40 * "=" + '\n\n'

    return emails



def pick_up_str_bigram(bigram, prev_word):
    """
    A function used to pick up a string randomly from the passed bigram
    :param bigram: a dictionary, the bigram model
    :param prev_word: a string, the previous single word that came before the next word that has to be predicted
    :return: 1) clause: a string, the predicted next word
             2) clause_pr: a float, the probability of the predicted word
    """

    # Building up the probability distribution for possible words that may come after prev_word
    pr_distr = list([k, bigram.get(k)] for k in bigram.keys() if k.split()[0] == prev_word)

    # Sorting the probability among the distribution table
    pr_distr.sort(key=operator.itemgetter(1))

    # Building the cumulative probabilities by summing the previous pr's
    for i in range(1, len(pr_distr)):
        new_value = pr_distr[i][1] + pr_distr[i - 1][1]
        pr_distr[i][1] = new_value

    # Pick up a random str of the form "<s>|I", "you|are" for example
    str_pr = random.uniform(0, 1)  # Randomly generated pr for the next word
    # The predicted next word 'token2' contained in a string of the form "token1|token2"
    clause, clause_pr = find_word_str_matches_pr(pr_distr, str_pr)

    return clause, clause_pr



def generate_bigram_emails(unsmoothed_bigram, smoothed_bigram):
    """
    A function used to generate emails using both unsmoothed and smoothed bigram models
    :param unsmoothed_bigram: a dictionary, the unsmoothed bigram model of the form {'token1|token2': pr, ...}
    :param smoothed_bigram: a dictionary, the smoothed bigram model of the form {'token1|token2': pr, ...}
    :return: emails: a string, the automatically generated emails by bigram models
    """

    end_sent_punct = ('!', '?', '.')  # End of sentence punctuations

    emails = "Emails generated using unsmoothed bigram:\n\n"

    # Generating 10 emails unsmoothed
    for email_count in range(10):
        emails += '%2d) ' % (email_count + 1)

        sentence_pr_log = 0.0  # automatically generated sentence's probability in log space

        # Generate a sentence
        words_count = random.randint(1, 30)  # words_count: A random number than does not exceed 30
        for i in range(words_count):
            # First, pick up a string that starts with <s>
            if i == 0:
                clause, clause_pr = pick_up_str_bigram(unsmoothed_bigram, "<s>")

            # If not first word to choose
            else:
                clause, clause_pr = pick_up_str_bigram(unsmoothed_bigram, clause.split()[1])

            # clause of the form 'token1 token2'. word is the next predicted token 'token2'
            word = clause.split()[1]
            sentence_pr_log += math.log2(clause_pr)
            # check if word is 'end of sentence punctuation' </s> then, stop
            if word == "</s>":
                break
            elif word in end_sent_punct:  # If end of sentence punctuation, print it then stop
                emails += ' ' + word + ' '
                break
            emails += ' ' + word + ' '
        emails += '[' + str(sentence_pr_log) + ']\n'

    emails += "\n\n" + 40 * "=" + '\n\n'
    emails += "Emails generated using smoothed bigram:\n"

    # Generating 10 emails smoothed
    for email_count in range(10):
        emails += '%2d) ' % (email_count + 1)

        sentence_pr_log = 0.0  # automatically generated sentence's probability in log space

        # Generate a sentence
        words_count = random.randint(1, 30)  # words_count: A random number than does not exceed 30
        for i in range(words_count):
            # First, pick up a string that starts with <s>
            if i == 0:
                clause, clause_pr = pick_up_str_bigram(smoothed_bigram, "<s>")

            # If not first word to choose
            else:
                clause, clause_pr = pick_up_str_bigram(smoothed_bigram, clause.split()[1])

            # clause of the form 'token1 token2'. word is the next predicted token 'token2'
            word = clause.split()[1]
            sentence_pr_log += math.log2(clause_pr)

            # check if word is 'end of sentence punctuation' </s> then, stop
            if word == "</s>":
                break
            elif word in end_sent_punct:  # If end of sentence punctuation, print it then stop
                emails += ' ' + word + ' '
                break
            emails += ' ' + word + ' '
        emails += '[' + str(sentence_pr_log) + ']\n'

    emails += "\n" + 40 * "=" + '\n\n'

    return emails



def pick_up_str_trigram(trigram, prev_2_words):
    """
    A function used to pick up a string randomly from the passed bigram
    :param trigram: a dictionary, the trigram model
    :param prev_2_words: a string, the previous two words that came before the next word that has to be predicted
    :return: 1) clause: a string, the predicted next word
             2) clause_pr: a float, the probability of the predicted word
    """
    # Building up the probability distribution for possible words that may come after prev_words
    pr_distr = list([k, trigram.get(k)] for k in trigram.keys() if k.startswith(prev_2_words))

    # Sorting the probability among the distribution table
    pr_distr.sort(key=operator.itemgetter(1))

    # Building the cumulative probabilities by summing the previous pr's
    for i in range(1, len(pr_distr)):
        new_value = pr_distr[i][1] + pr_distr[i - 1][1]
        pr_distr[i][1] = new_value

    # Pick up a random word from strings of the form "<s> <s>| ", "you are| amous" for example
    str_pr = random.uniform(0, 1)  # Randomly generated pr for the next word
    # The predicted next word 'token3' contained in a string of the form "token1 token2 token3"
    clause, clause_pr = find_word_str_matches_pr(pr_distr, str_pr)

    return clause, clause_pr



def generate_trigram_emails(unsmoothed_trigram, smoothed_trigram):
    """
    A function used to generate emails using both unsmoothed and smoothed trigram models
    :param unsmoothed_trigram: a dictionary, the unsmoothed trigram model of the form {'token1|token2|token3': pr, ...}
    :param smoothed_trigram: a dictionary, the smoothed trigram model of the form {'token1|token2|token3': pr, ...}
    :return: emails: a string, the automatically generated emails by bigram models
    """

    end_sent_punct = ('!', '?', '.')  # End of sentence punctuations
    emails = "Emails generated using unsmoothed trigram:\n\n"

    # Generating 10 emails unsmoothed
    for email_count in range(10):
        emails += '%2d) ' % (email_count + 1)

        sentence_pr_log = 0.0  # automatically generated sentence's probability in log space

        # Generate a sentence
        words_count = random.randint(1, 30)  # words_count: A random number than does not exceed 30
        for i in range(words_count):
            # First, pick up a word that comes at the start of sentences
            if i == 0:
                clause, clause_pr = pick_up_str_trigram(unsmoothed_trigram, "<s> <s>")

            # If not first word to choose
            else:
                prev_2_words = clause.split()[1] + ' ' + clause.split()[2]
                clause, clause_pr = pick_up_str_trigram(unsmoothed_trigram, prev_2_words)

            # str of the form 'token1 token2 token3'. word is the next predicted token 'token3'
            word = clause.split()[2]
            sentence_pr_log += math.log2(clause_pr)

            # check if word is 'end of sentence punctuation' </s> then, stop
            if word == "</s>":
                break
            elif word in end_sent_punct:  # If end of sentence punctuation, print it then stop
                emails += ' ' + word + ' '
                break
            emails += ' ' + word + ' '
        emails += '[' + str(sentence_pr_log) + ']\n'

    emails += "\n\n" + 40 * "=" + '\n\n'
    emails += "Emails generated using smoothed trigram:\n"

    # Generating 10 emails smoothed
    for email_count in range(10):
        emails += '%2d) ' % (email_count + 1)

        sentence_pr_log = 0.0  # automatically generated sentence's probability in log space

        # Generate a sentence
        words_count = random.randint(1, 30)  # words_count: A random number than does not exceed 30
        for i in range(words_count):
            # First, pick up a word that comes at the start of sentences
            if i == 0:
                clause, clause_pr = pick_up_str_trigram(smoothed_trigram, "<s> <s>")

            # If not first word to choose
            else:
                prev_2_words = clause.split()[1] + ' ' + clause.split()[2]
                clause, clause_pr = pick_up_str_trigram(smoothed_trigram, prev_2_words)

            # str of the form 'token1 token2 token3'. word is the next predicted token 'token3'
            word = clause.split()[2]
            sentence_pr_log += math.log2(clause_pr)

            # check if word is 'end of sentence punctuation' </s> then, stop
            if word == "</s>":
                break
            elif word in end_sent_punct:  # If end of sentence punctuation, print it then stop
                emails += ' ' + word + ' '
                break
            emails += ' ' + word + ' '
        emails += '[' + str(sentence_pr_log) + ']\n'

    emails += "\n" + 40 * "=" + '\n\n'

    return emails


def estimate_pr_pp_smoothed_trigram(test_sentences, smoothed_trigram, freq_bigram, bigram_v, start_end_punct_count):
    '''
    A function used to calculated the probability of each sentence of teh test data set
    :param test_sentences: a list, the sentences that we calculate the probability for
    :param smoothed_trigram: a dictionary, the smoothed trigram model
    :param freq_bigram: a dictionary, the frequency bigram model 'frequency of each sequence words'
    :param start_end_punct_count: an integer, the count of <s> or </s>
    :return: 1) pr_ans: a string, the probabilities of the test sentences
             2) pp_ans: a string, the perplexities of the test sentences
    '''

    # probabilities of sentences
    pr_ans = "Estimating probabilities of sentences using smoothed trigram in log space:\n\n"

    # buffer to output the perplexities
    pp_ans = "Evaluating perplexities of sentences using smoothed trigram are:\n\n"

    # for each line of the test data set
    for line in test_sentences:
        # Tokenize each line by white spaces
        tokens = line.split()
        tokens_n = len(tokens)      # number of tokens in each sentence

        # Check if the length of the line is zero then skip it
        if tokens_n == 0:
            continue

        pp = 1.0                # perplexities of sentences of smoothed trigram
        pr_log = 0.0                    # probabilities of sentences of smoothed trigram in log space

        # In each line
        for i in range(2, tokens_n):
            clause = tokens[i - 2] + ' ' + tokens[i - 1] + ' ' + tokens[i]
            # Check if clause is in smoothed trigram keys then, takes its probability else, its probability is zero
            if clause in smoothed_trigram.keys():
                pr_log += math.log2(smoothed_trigram[clause])
            elif clause.startswith('<s> <s>'):
                # If the string starts with '<s> <s>' and not in trigram 'frequency' model then its probability is zero
                # added to one 'smoothing' divided by '<s> <s>' count added to vocabulary size
                pr_log += math.log2(((0 + 1) / (start_end_punct_count + bigram_v)))
            else:
                # Check if the first two tokens exist in the trained bigram model if so, get the
                # frequency of 'token1 token2' else, it's zero then smooth it
                first_2_tokens = clause.split()[0] + ' ' + clause.split()[1]
                if first_2_tokens in freq_bigram.keys():
                    pr_log += math.log2((0 + 1) / (freq_bigram[first_2_tokens] + bigram_v))
                else:
                    pr_log += math.log2((0 + 1) / (0 + bigram_v))

        pp = (2 ** (-1 * (1 / tokens_n) * (pr_log)))

        # Add the answers to the string
        pp_ans += ' [' + str(pp) + '], '
        pr_ans += ' [' + str(pr_log) + '], '

    pr_ans += "\n\n" + 40 * "=" + '\n\n'
    pp_ans += "\n\n" + 40 * "=" + '\n\n'

    return pr_ans, pp_ans


def evaluate_pp_smoothed_brigram(test_sentences, smoothed_bigram, freq_unigram, unigram_v, start_end_punct_count):
    '''
    A function used to calculated the probability of each sentence of teh test data set
    :param test_sentences: a list, the sentences that we calculate the probability for
    :param smoothed_bigram: a dictionary, the smoothed trigram model
    :param freq_unigram: a dictionary, the frequency bigram model 'frequency of each sequence words'
    :param start_end_punct_count: an integer, the count of <s> or </s>
    :return: pp_ans: a string, the perplexities of the test sentences
    '''

    # buffer to output the perplexities
    pp_ans = "Evaluating perplexities of sentences using smoothed bigram:\n\n"

    # for each line of the test data set
    for line in test_sentences:
        # Tokenize each line by white spaces
        tokens = line.split()
        tokens_n = len(tokens)      # number of tokens in each sentence

        # Check if the length of the line is zero then skip it
        if tokens_n == 0:
            continue

        pp = 1.0                        # perplexities of sentences of smoothed bigram
        pr_log = 0.0                    # probabilities of sentences of smoothed bigram in log space

        # In each line
        for i in range(1, tokens_n):
            clause = tokens[i - 1] + ' ' + tokens[i]
            # Check if clause is in smoothed bigram keys then, takes its probability else, its probability is zero
            if clause in smoothed_bigram.keys():
                pr_log += math.log2(smoothed_bigram[clause])
            elif clause.startswith('<s>'):
                # If the string starts with '<s>' and not in bigram 'frequency' model then its probability is zero
                # added to one 'smoothing' divided by '<s>' count added to vocabulary size
                pr_log += math.log2(((0 + 1) / (start_end_punct_count + unigram_v)))
            else:
                # Check if the first two tokens exist in the trained unigram model if so, get the
                # frequency of 'token1' else, it's zero
                prev_token = clause.split()[0]
                if prev_token in freq_unigram.keys():
                    pr_log += math.log2((0 + 1) / (freq_unigram[prev_token] + unigram_v))
                else:
                    pr_log += math.log2((0 + 1) / (0 + unigram_v))

        pp = (2 ** (-1 * (1 / tokens_n) * (pr_log)))

        # Add the answers to the string
        pp_ans += ' [' + str(pp) + '], '

    pp_ans += "\n\n" + 40 * "=" + '\n\n'

    return pp_ans


"""
text: The original input text (both train and test)
train_text: Train text (%60 of the text) 
test_text: Test text (%40 of the text)
"""

# Start timing
start_time = time.time()

# Reading the text file and divide the text into (%60 of the text) train_text, and (%40) Test_text.
train_text, test_text = read_text_file()

# Tokenize the text or splitting by white spaces
train_tokens = train_text.split()

# Converting the tokens from a string into a list with lower case tokens
train_tokens = [word.lower() for word in train_tokens]

# Implementing Unigram, Bigram, Trigram frequency or number of occurence models and storing
# 'N-Gram_v': The vocabulary size, and 'N-Gram_n': Tokens count into variables
unigram_freq_model, unigram_v, unigram_n = build_unigram_freq_model(train_tokens)
bigram_freq_model, tokens_freq_bigram, bigram_v, bigram_n = build_bigram_freq_model(train_tokens)

trigram_freq_model, tokens_freq_trigram, trigram_v, trigram_n, start_end_punct_count = \
    build_trigram_freq_model(train_tokens)


# Implementing smoothed and unsmoothed models and their probabilities of the form [[str, pr] ...]
unsmoothed_unigram, smoothed_unigram = build_pr_unigram(unigram_freq_model, unigram_v, unigram_n)
unsmoothed_bigram, smoothed_bigram = build_pr_bigram(bigram_freq_model, tokens_freq_bigram, unigram_v)
unsmoothed_trigram, smoothed_trigram = build_pr_trigram(trigram_freq_model, bigram_freq_model,
                                                        bigram_v, start_end_punct_count)

# Implementing probability distribution in order to generate emails
pr_distr_unsmoothed_unigram = build_pr_distr_unigram(unsmoothed_unigram)
pr_distr_smoothed_unigram = build_pr_distr_unigram(smoothed_unigram)

emails_unigram = generate_unigram_emails(pr_distr_unsmoothed_unigram, pr_distr_unsmoothed_unigram)
emails_bigram = generate_bigram_emails(unsmoothed_bigram, smoothed_bigram)
emails_trigram = generate_trigram_emails(unsmoothed_trigram, smoothed_trigram)

# Tokenize the text or splitting by white spaces
test_tokens = test_text.split()

# Adding punctuation start and end of sentence and separate lines by '\n'
test_sentences = ""
for token in test_tokens:
    if token == "<s>":
        test_sentences += ' <s> <s> '
    elif token == "</s>":
        test_sentences += ' </s> </s>\n'
    else:
        test_sentences += ' ' + token.lower() + ' '
test_sentences = test_sentences.split('\n')

# Estimating the probabilities and evaluating the perplexities of test data set sentences according
# to smoothed trigram

estimated_pr_trigram, evaluating_pp_trigram = estimate_pr_pp_smoothed_trigram(test_sentences, smoothed_trigram,
                                                                              bigram_freq_model,bigram_v,
                                                                              start_end_punct_count)

# Evaluating smoothed bigram model's performance using perplexity
evaluating_pp_bigram = evaluate_pp_smoothed_brigram(test_sentences, smoothed_bigram, unigram_freq_model,
                                                    unigram_v, start_end_punct_count)



# End timing
end_time = time.time()

elapsed_time = end_time - start_time
elapsed_time_str = "\n\nThe elapsed time is: [%.5f] Sec or [%.5f] Min\n" % (elapsed_time, elapsed_time / 60)

# Writing the output data on results.txt file
delete_out_file()
out_on_file(emails_unigram + emails_bigram + emails_trigram + estimated_pr_trigram + evaluating_pp_trigram
            + evaluating_pp_bigram + elapsed_time_str)

print(elapsed_time_str)
