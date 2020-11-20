import util
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.naive_bayes import MultinomialNB
import gensim.downloader as api
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
from gensim.models import KeyedVectors
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

def main(folder_path, train_file, test_file, save_file):

    # train_X, train_y = util.load_dataset(folder_path, train_file)
    # test_X, test_y = util.load_dataset(folder_path, test_file)

    # lrm = NaiveBayesModel()
    # lrm.fit(train_X, train_y, (1, 1))
    # pred_y = lrm.predict(test_X)
    #
    # util.print_accuracy_measures(test_y, pred_y)

    train_X = [
        'Now that Donald Trump is the presumptive GOP nominee, it s time to remember all those other candidates who tried so hard to beat him in the race to the White House. After all, how can we forget all the missteps, gaffes, weirdness, and sheer idiocies of such candidates as Jeb Bush, Marco Rubio, John Kasich, Ted Cruz, Ben Carson, and Carly Fiorina?There s a video making the rounds on Twitter that does just that, and eulogizes three of these failed candidates as though they re dead (and the GOP itself might as well be dead at this point anyway). Appropriately titled,  A Eulogy for the GOP,  people make short speeches about each of these candidates.Once past the man who actually says Jeb Bush was qualified to be president, there are fake tears, with journalist and comedienne Francesca Fiorentini saying: Dearly beloved, we re gathered here today to commemorate the candidates that are no longer with us. One man, speaking to an amusing, circus-y rendition of Chopin s Funeral March, remembers Jeb this way: Jeb, we hardly knew ye. It s too bad that your policies couldn t find their way out of the Republican morass. Indeed, it was like Jeb and his policies were invisible sometimes, although many felt that he would win the nomination early on. His best performances often came during debates, when he d get into out-and-out fights with Donald Trump over virtually everything under the sun. He also had a bad habit of defending his brother s actions in Iraq, memorably saying,  As it relates to my brother, there s one thing I know for sure: He kept us safe. Moving on to Marco Rubio, Fiorentini herself says: He was called so many things:  Young, charming, Lil  Marco.\' That last nickname is, of course, a reference to Trump s penchant for name-calling. Then, a man speaking Spanish remembers Rubio this way: Marco, why aren t you a normal boy? Another said that his involvement in the Gang of Eight immigration bill killed him as a candidate, to which Fiorentini replies: Some people die of gang violence. He died of Gang of Eight violence. The makers of this video saved the absolute best for last, of course, which is Ben Carson. On our dear Dr. Carson, one of the speakers says: He would gently rock me to sleep with his monotone voice. Carson was especially well known for seemingly being asleep half the time. If we saw his eyes open wide, it was surprising, and likely because he was surprised himself. His voice is soft, his speech is slow, and it really can be hard to stay awake while he s talking. Imagine him giving a State of the Union address!To see the whole video, especially the spectacular ending after Carson s eulogy, watch below:Take a moment to say your g byes to these GOP candidates. ? https://t.co/6O70bl9zV8  AJ+ (@ajplus) May 7, 2016 Featured image via screen capture from embedded video',
        'DSAN FRANCISCO (Reuters) - California Attorney General Xavier Becerra said on Friday he was â€œprepared to take whatever action it takesâ€ to defend the Obamacare mandate that health insurers provide birth control, now that the Trump administration has moved to circumvent it. The administrationâ€™s new contraception exemptions â€œare another example of the Trump administration trampling on peopleâ€™s rights, but in this case only women,â€ Becerra told Reuters.  Becerra and other Democratic attorneys general have filed courtroom challenges to other Trump administration policies involving healthcare, immigration and the environment.',
        'WASHINGTON (Reuters) - As a lawyer in private practice for a decade, President Donald Trumpâ€™s U.S. Supreme Court nominee Neil Gorsuch often fought on behalf of business interests, including efforts to curb securities class action lawsuits, experience that could mould his thinking if he is confirmed as a justice. Gorsuch, a conservative federal appeals court judge from Colorado nominated by wealthy businessman Trump on Tuesday, could turn out to be a friend to business, having represented the U.S. Chamber of Commerce in fending off securities class actions, one of the most hotly contested areas of corporate law. The chamber is the largest U.S. business lobbying group. If confirmed, Gorsuch would be one of the only current justices with extensive experience on business issues in private practice.  Securities class action lawsuits are filed by investors who allege misconduct by a company whose stock price has tanked, hurting investorsâ€™ portfolios. These once-common lawsuits now face higher hurdles and are filed less often. Congress passed laws in 1995 and 1998 making it harder to bring securities class actions. Later court rulings, including one in which Gorsuch was involved, clarified the requirements under the laws. From 1995 to 2005, Gorsuch worked at boutique law firm Kellogg, Huber, Hansen, Todd, Evans & Figel in Washington, becoming a partner in 1998. He had a wide range of clients, including individuals and nonprofits, as well as various business interests. While there, he filed two briefs on behalf of the Chamber of Commerce seeking limits to securities class actions. One of Gorsuchâ€™s briefs came in a securities fraud case called Dura Pharmaceuticals v. Broudo. The Supreme Court in 2005 ruled unanimously for the company, but did not issue the kind of broad ruling Gorsuch had sought, said Patrick Coughlin, the lawyer who argued the case on behalf of the investors who sued. Of Gorsuchâ€™s role, Coughlin said representing the Chamber of Commerce is the epitome of corporate defense work for a lawyer. â€œThe chamber had always been against us,â€ Coughlin said, referring to class action lawyers. â€œHeâ€™d always been on the other side of what we were doing, so it was not surprising he was selected by Trump.â€ Prior to the ruling, Gorsuch co-wrote an article in the Legal Times trade publication for lawyers in which he described some securities class actions as a â€œfree ride to fast richesâ€ for plaintiffsâ€™ lawyers. He said the Dura case was a chance for the court to â€œcurb frivolous fraud claimsâ€ in which plaintiffs cannot prove a stock price drop was caused by misrepresentations by a company. Gorsuchâ€™s background promises to be valuable on the Supreme Court, which hears a significant number of business disputes among the roughly 70 cases considered each annual term.  Businesses have been trying with mixed success to get the Supreme Court to put new curbs on class action litigation beyond the securities context. Class actions can lead to huge jury awards against companies and is harder to defend against than lawsuits brought by individuals. If confirmed by the U.S. Senate in time, Gorsuch could play an immediate role in a major case on whether companies can head off costly class action lawsuits by forcing employees to give up their right to pursue work-related legal claims in court as a group. Paul Bland, executive director of consumer advocacy group Public Justice, said he hoped Gorsuch, if confirmed, would see that â€œmost of what he saw as meritless cases in 2005 have been weeded out, and that the vast majority of the cases that remain raise substantial issues that protect investors.â€ Chamber of Commerce official Tom Collamore joined Trump at the White House on Wednesday for a meeting with advocacy groups touting Gorsuchâ€™s pick, calling it â€œa fantastic nomination.â€ The chamber declined comment on Gorsuchâ€™s work for the group. In a statement issued after the nomination was announced, chamber President Tom Donohue congratulated Trump on the selection. When Democratic President Barack Obama last March nominated appeals court judge Merrick Garland to fill the same vacant seat on the court, there was no press release from the chamber. Senate Republicans refused to act on Garlandâ€™s nomination, a move that paved the way for Trump to nominate Gorsuch to replace fellow conservative Antonin Scalia, who died in February 2016. Describing his career in private practice, Gorsuch said in his Senate questionnaire prior to his appointment by Republican President George W. Bush to the Denver-based 10th U.S. Circuit Court of Appeals in 2006 that he was â€œinvolved in matters large and small for clients ranging from individuals to nonprofits to corporations,â€ on such issues as racketeering, securities fraud and antitrust.  His former Washington law firm stressed in a statement that Gorsuch had a â€œwide varietyâ€ of clients and cited a case in which he represented people who had been targeted over payday loans. David Frederick, a lawyer at the firm, called Gorsuch â€œa dogged, very determined lawyer.â€ â€œHeâ€™s the kind of lawyer you would want to have representing you,â€ Frederick said. After leaving the firm, but before becoming a judge, he spent just over a year in Bushâ€™s Justice Department. One of his roles was overseeing antitrust litigation involving the government. '
    ]

    train_y = [0, 1, 0]

    # trying out https://keras.io/examples/nlp/pretrained_word_embeddings/

    vectorizer = TextVectorization()
    vectorizer.adapt(train_X)

    vocab = vectorizer.get_vocabulary()

    word_index = dict(zip(vocab, range(len(vocab))))

    # import pretrained model
    info = api.info()
    # see if this is taking up time
    word_emb_model = api.load("word2vec-google-news-300")
    # word_emb_model = KeyedVectors.load_word2vec_format("word2vec-google-news-300")

    # simple NumPy matrix where entry at index i is the pre-trained vector for the word of index i in our vectorizer's vocabulary
    # embedding matrix
    num_tokens = len(vocab) + 2
    embedding_dim = 300

    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():

        if word in word_emb_model.vocab:
            embedding_matrix[i] = word_emb_model[word]

    a = 1


    # tfidf = TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS)
    # new_X = tfidf.fit_transform(train_X)
    # print(new_X)
    # b = tfidf.fit(train_X)
    #
    # model = MultinomialNB()
    # model.fit(new_X, train_y)
    #
    # vectors = []
    #
    # for msg in train_X:
    #     cleaned_text = msg.lower()
    #     cleaned_text = remove_stopwords(cleaned_text)
    #     cleaned_text_arr = preprocess_string(cleaned_text)
    #
    #     vector = [word_emb_model[word] for word in cleaned_text_arr
    #               if word in word_emb_model.vocab]
    #     vectors.append(vector)
    #
    # print(vectors)
    #
    # model = MultinomialNB()
    # model.fit(vectors, train_y)

    # pl = Pipeline([('wdemb', word_emb_model),
    #                ('nb', MultinomialNB())])
    # model = pl.fit(train_X, train_y)
    # y_pred = model.predict(test_X)
    # util.print_accuracy_measures(test_y, y_pred)

class NaiveBayesModel:
    """
    Example usage:
        > clf = NaiveBayesModel()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self):
        """
        """
        self.model = None

    def fit(self, X, y, ngram=(1, 1)):
        pl = Pipeline([('tfidf', TfidfVectorizer(stop_words=text.ENGLISH_STOP_WORDS, ngram_range=ngram)),
                       ('nb', MultinomialNB())])
        self.model = pl.fit(X, y)

    def predict(self, X):
        lr_pred = self.model.predict(X)

        return lr_pred

if __name__ == "__main__":
    folder_path = 'datasets/kaggle_comp/split_files/'
    train_file = 'train.csv'
    test_file = 'test.csv'
    save_file = 'test_pred.csv'

    main(folder_path, train_file, test_file, save_file)
