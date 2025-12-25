from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

def earnings_report_nlp_analysis():
    """
        Perform NLP analysis on earnings call transcripts.
        Identify spikes in negative phrases that correlate with stock price drops.
    """
    with open("AAPL_2024Q1.txt", "r", encoding="utf-8") as f:
        raw = f.read()


    def clean_earningscall(text):
        # Remove "speaker ..." labels entirely
        text = re.sub(r"\bspeaker[\s\.\w]*?(?=\b[a-z]|$)", " ", text, flags=re.IGNORECASE)

        # Remove job titles and company names after speaker lines
        text = re.sub(r"\b(ceo|cfo|analyst|operator|director of investor relations|at|from)\b", "", text, flags=re.IGNORECASE)

        # Remove leftover periods between single words (like "apple. vision. pro.")
        text = re.sub(r"\b([a-zA-Z])\.\s(?=[a-zA-Z])", r"\1 ", text)
        text = re.sub(r"(\b\w+)\.\s(\w+\b)", r"\1 \2", text)

        # Collapse repeated periods or single-letter dots
        text = re.sub(r"\s*\.\s*", ". ", text)
        text = re.sub(r"\.{2,}", ".", text)

        # Clean unwanted symbols and extra spaces
        text = re.sub(r"[^a-zA-Z0-9,\.\?\!\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text
    with open("AAPL_2024Q1.txt", "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_earningscall(raw)

    with open("AAPL_2024Q1_cleaned_fixed.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)


    nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(cleaned)
    print(score)

    sentences = cleaned.split(".")
    sentences_clean = [s.strip() for s in sentences if s.strip()]
    scores = [sia.polarity_scores(s)["compound"] for s in sentences if s.strip()]
    print(len(sentences))

    sentence_scored_df = pd.DataFrame({"sentence": sentences_clean, "score": scores})
    #sent_df.to_csv("report.csv", index = False)
    print("Min sentiment:", sentence_scored_df["score"].min())

    """
        These two metrics (avg_sent, neg_ratio) are what i can later merge
        into my earnings_df by stock/quarter and compare with ret_3d_from_earnings.
    """

    neg_ratio = (sentence_scored_df["score"] < -0.2).mean()
    avg_sent = sentence_scored_df["score"].mean()
    print(f"Average sentiment: {avg_sent:.3f}")
    print(f"Percent negative sentences: {neg_ratio*100:.1f}%")

    return sentence_scored_df

