from langchain.prompts import PromptTemplate

class PromptManager:
    @staticmethod
    def get_summary_prompts(media_type: str):
        if media_type == "twitter_space":
            summary_template = """
            You are an analytics professional at Lido Finance, a Liquid Staking protocol for Ethereum. You are given a transcript of Twitter Spaces in the crypto/web3 space that may or may not be related to Lido.
            Given the transcript, you are writing structured notes in markdown format. Think of your notes as key takeaways, TLDRs, and executive summaries.

            Your notes should be concise, detailed, and structured by topics. You know what information is especially important, and what is less important.

            Here is the transcript:
            {text}
            
            YOUR NOTES:
            """
            refine_template = """
            You are an analytics professional at Lido Finance, a Liquid Staking protocol for Ethereum. You are given a transcript of Twitter Spaces in the crypto/web3 space that may or may not be related to Lido.
            Given the transcript, you are refining structured notes in markdown format. Think of your notes as key takeaways, TLDRs, and executive summaries.

            Here is the existing note:
            {existing_answer}
            
            We have the opportunity to refine the existing note (only if needed) with some more context below:
            -----
            {text}
            -----
            
            Given the new context, refine the original note to make it more complete.
            If the context isn't useful, return the original summary.

            Your notes should be concise, detailed, and structured by topics. You know what information is especially important, and what is less important.

            Use markdown formatting to its fullest to produce visually appealing, structured notes.
            """
        elif media_type == "youtube":
            summary_template = """
            You are an analytics professional at Lido Finance, a Liquid Staking protocol for Ethereum. You are given a transcript of a YouTube video related to Crypto/Web3 that may or may not be related to Lido.
            You are writing structured notes in markdown format. Think of your notes as key takeaways, TLDRs, and executive summaries.

            Your notes should be concise, detailed, and structured by topics. You know what information is especially important, and what is less important.

            Here is the transcript:
            {text}

            YOUR NOTES:
            """
            refine_template = """
            You are refining structured notes in markdown format for Lido Finance. Think of your notes as key takeaways, TLDRs, and executive summaries.

            Here is the existing note:
            {existing_answer}

            We have the opportunity to refine the existing note with some more context below:
            -----
            {text}
            -----

            Refine the original note with the new context. If the new context isn't useful, return the original summary.

            Your notes should be concise, detailed, and structured by topics.
            """
        elif media_type == "governance_forum_summary":
            summary_template, refine_template = PromptManager.get_governance_forum_summary_prompts()
        elif media_type == "twitter_digest":
            summary_template = """
            You are an analytics professional at Lido, a Liquid Staking protocol for Ethereum.
            You have multiple tweets from X (Twitter). Summarize them in markdown, highlighting references to
            Lido, LSTs, Ethereum staking, the ethereum ecosystem at large, or crypto in general if they exist. Organize key points clearly.

            IMPORTANT. Your output must be of 250 characters or less.

            Tweets:
            {text}

            YOUR SUMMARY:
            """
            refine_template = """
            You are refining a summary of tweets for Lido context. Only refine if there's new relevant info.
            Existing summary:
            {existing_answer}

            Additional tweets:
            {text}

            IMPORTANT. Your output must be of 400 characters or less.
            """
            return (
                PromptTemplate.from_template(summary_template),
                PromptTemplate.from_template(refine_template)
            )
        elif media_type == "tweet_or_thread":
            summary_template = """
            You are an analyst tasked with summarizing a tweet or a tweet thread.
            Focus on extracting the key information, context, and main ideas expressed in the original tweets.
            
            Text:
            {text}
            
            YOUR NOTES:
            """
            refine_template = """
            Refine the summary for the main tweet or tweet thread.
            Existing summary:
            {existing_answer}
            
            Additional context:
            {text}
            """
            return (
                PromptTemplate.from_template(summary_template),
                PromptTemplate.from_template(refine_template)
            )
        elif media_type == "tweet_comments":
            summary_template = """
            You are an analyst tasked with summarizing the comments (replies) to a tweet or tweet thread.
            Focus on identifying the overall sentiment, key points raised by the community, and any notable reactions.
            
            Comments:
            {text}
            
            YOUR NOTES:
            """
            refine_template = """
            Refine the summary for the tweet comments.
            Existing summary:
            {existing_answer}
            
            Additional comments:
            {text}
            """
            return (
                PromptTemplate.from_template(summary_template),
                PromptTemplate.from_template(refine_template)
            )
        else:
            # Article
            summary_template = """
            You are an analytics professional at Lido Finance, a leading Liquid Staking protocol for Ethereum. 
            Your task is to summarize content related to Lido, DeFi, staking, or the broader Ethereum ecosystem. 
            The content provided could be an article, a tweet or tweet thread, or a YouTube video transcript.

            Analyze the provided content and extract key takeaways, TLDRs, and executive summaries in a structured markdown format.
            Ensure that your summary is clear, concise, and useful for decision-makers.

            Content:
            {text}

            YOUR NOTES:
            """
            refine_template = """
            You are an analytics professional at Lido Finance, specializing in Liquid Staking and the Ethereum ecosystem. 
            You are given a transcript of content (which could be an article, a tweet or tweet thread, or a YouTube video transcript). 
            Your goal is to refine and improve structured notes in markdown format, ensuring clarity, accuracy, and completeness.

            Consider the notes as key takeaways, TLDRs, and executive summaries.

            Here is the existing note:

            {existing_answer}

            New context:
            {text}

            Refine the notes based on the new information while maintaining structure and coherence.
            """
        return (PromptTemplate.from_template(summary_template),
                PromptTemplate.from_template(refine_template))

    @staticmethod
    def get_executive_prompts(media_type: str):
        if media_type == "twitter_space":
            exec_template = """
             Given the summary of a Twitter Space:
            {text}

            Generate an extremely brief executive summary for Lido contributors. It should be concise, focused, and only contain information relevant to Lido Finance.
            """
            refine_exec_template = """
            You are an analytics professional at Lido Finance, a Liquid Staking protocol for Ethereum. You are given a summary of Twitter Spaces in the crypto/web3 space that may or may not be related to Lido.
            Given the summary, you are refining an executive summary in markdown format. Think of your notes as key takeaways, TLDRs, and executive summaries.

            Here is the existing executive summary:
            {existing_answer}

            We have the opportunity to refine the existing executive summary (only if needed) with some more context below:
            -----
            {text}
            -----

            Your updated executive summary:
            """
        elif media_type == "youtube":
            exec_template = """
            Given the summary of a YouTube video:
            {text}

            Generate an extremely brief executive summary for Lido contributors. It should be concise, focused, and only contain information relevant to Lido Finance.
            """
            refine_exec_template = """
            Refine an executive summary in markdown format based on the following summary:

            Existing summary:
            {existing_answer}

            New context:
            -----
            {text}
            -----

            Your refined executive summary:
            """
        elif media_type == "governance_forum":
            exec_template, refine_exec_template = PromptManager.get_governance_forum_executive_prompts()
        elif media_type == "twitter_digest":
            exec_template = """
            You have a summary of multiple tweets which may or may not be related to crypto/web3, but they are from Twitter accounts in the space. Focus on Lido, LST and Ethereum references.
            {text}

            Given the summary, you are refining an executive summary in markdown format. Think of your notes as key takeaways, TLDRs, and executive summaries.

            IMPORTANT. Your output must be of 250 characters or less.

            """
            refine_exec_template = """
            Refine the existing Twitter digest executive summary if new text is relevant. Otherwise return original.
            Existing summary:
            {existing_answer}

            Additional snippet:
            {text}

            IMPORTANT. Your output must be of around 300 characters or less.
            """
            return (
                PromptTemplate.from_template(exec_template),
                PromptTemplate.from_template(refine_exec_template)
            )
        else:
            # Default: Article
            exec_template = """
            Given the summary of an Article:
            {text}

            Generate an extremely brief executive summary for Lido contributors. It should be concise, focused, and, if available, contain information relevant to Lido. 
            """
            refine_exec_template = """
            Refine an executive summary in markdown format based on the following article summary:

            Existing summary:
            {existing_answer}

            New context:
            -----
            {text}
            -----

            Your refined executive summary:
            """
        return (PromptTemplate.from_template(exec_template),
                PromptTemplate.from_template(refine_exec_template))
    
    @staticmethod
    def get_governance_forum_summary_prompts():
        summary_template = """
        You are an analytics professional at Lido Finance, a Liquid Staking protocol for Ethereum. 
        You are given content from governance forums in the crypto/web3 space that may or may not be related to Lido.
        Given the transcript, you are writing structured notes in markdown format. Think of your notes as key takeaways, TLDRs, and executive summaries.
        Your notes should be concise, detailed, and structured by topics, but without starting with the forum name. 
        You know what information is especially important, and what is less important.

        IMPORTANT. Your output must be of 300 characters or less.

        Here is the transcript:
        {text}
        
        YOUR NOTES:
        """

        refine_template = """
        You are refining structured notes in markdown format for Lido Finance. Think of your notes as key takeaways, TLDRs, and executive summaries.

        Here is the existing note:
        {existing_answer}
        
        We have the opportunity to refine the existing note with some more context below:
        {text}
        
        Refine only if the new context is relevant.
        If not relevant, return the original summary.
        
        IMPORTANT. Your output must be of 250 characters or less.
        
        Keep the notes concise and structure, while not going over 250 characters.
        """
        return summary_template, refine_template

    @staticmethod
    def get_governance_forum_executive_prompts():
        exec_template = """
        Given the summarized governance forum discussions:
        {text}

        Generate an extremely brief executive summary suitable for Lido contributors. 
        It should be concise, focused on key governance changes or proposals that matter to Lido or Ethereum staking, 
        and highlight any immediate actions or decisions.

        IMPORTANT. Your output must be of 250 characters or less.

        """

        refine_exec_template = """
        Refining the executive summary for governance updates.
        Existing summary:
        {existing_answer}

        New context:
        {text}

        Update the executive summary only if it improves clarity or relevance to Lido. Otherwise, return the original.
        IMPORTANT. Your output must be of 250 characters or less.
        """
        return exec_template, refine_exec_template

    @staticmethod
    def get_governance_forum_relevance_prompt():
        return (
            "You are an assistant. You are given the entire text of a governance forum topic, which may include the title, "
            "an excerpt or summary, and the main body of the post.\n\n"
            "Determine if this topic is relevant to Lido (a liquid staking protocol), its ecosystem, stETH, wstETH, "
            "LSDs, Ethereum staking, or competitor LSD protocols. Also consider if it can indirectly impact Lido's strategy, "
            "treasury, integrations, or governance.\n\n"
            "Answer 'yes' if relevant. Otherwise, answer 'no'."
        )
    
    @staticmethod
    def get_twitter_relevance_prompt():
        """
        Returns the short text we will feed into GPT to check if a tweet is relevant
        to Lido, LSD protocols, stETH, wstETH, or Ethereum.
        """
        return (
            "You are an assistant. You are given the entire text of a tweet."
            "Determine if this topic is relevant to Lido (a liquid staking protocol), its ecosystem, stETH, wstETH, "
            "LSDs, Ethereum staking and Ethereum's ecosystem, or competitor LSD protocols. Also consider if it can indirectly impact Lido's strategy, "
            "treasury, integrations, or governance.\n\n"
            "Answer 'yes' if relevant. Otherwise, answer 'no'."
        )
    
    @staticmethod
    def get_governance_forum_category_prompt():
        """
        This prompt instructs GPT to pick exactly one category for each governance topic.
        """
        return (
        "You are given a governance forum post's full text."
        "You must assign exactly ONE category from this fixed list:"

        "1) 'Lido Related Updates 💧' - topics directly from the Lido Research Forum at research.lido.fi\n"
        "2) 'Competitor Updates 🥊' - these are usually topics from competitors such as Stader Labs, EtherFi, RocketPool, Karak, Frax, Chaos Labs, EigenLayer"
        "or simply topics on other forums that are prompting some competing protocols\n"
        "3) 'Lending Markets 🏦' - topics from lending protocols such as AAVE, Compound, Morpho, Venus, ListaDAO, MakerDAO, Moonwell or related to token lending, borrowing or similar topics\n"
        "4) 'Layer2s 🔗' - from forums such as Polygon, Arbitrum, Optimism or other layer twos or updates around layer 2s\n"
        "5) 'DEXes 💱' - topics related to decentralized exchanges such as Balancer, GMX, Curve, Sushiswap, Uniswap, dYdX or similar discussions around decentralized trading\n"
        "6) 'Grants and Funding 💸' \n"
        "7) 'Misc. 🌀' - topics that do not fit in the other categories or interesting news overall\n"
        "Return your answer exactly as the category name (including the emoji if present). \n"
        "Try to be certain on the category, otherwise pick something really close, or default to 'Misc. 🌀'.\n"
       "Your output must be exactly one line, with only the category text (including the emoji) and nothing else."
        )
    

    @staticmethod
    def get_telegram_format_prompt() -> PromptTemplate:
        """
        Returns a prompt template that instructs OpenAI to convert a Discord-formatted message
        into Telegram-friendly Markdown formatting.
        """
        template = (
            "The following text is formatted for Discord. Convert it into Telegram HTML. "
            "Preserve all the content but adjust any formatting including, but not limited to bolding, italics, headlines, and hyperlinks added properly on keywords and not in square brackets"
            "[] so they render properly in Telegram HTML, using the tags. "
            "Replace words wrapped by stars * or **, or words that start with one or multiple hashtags # with the words in bold formatted for telegram."
            "Do not alter any content; only change the formatting to create lists, but keep numbered items if they already exist, and transform headlines into bold tags wrapped content."
            "Also, keep the newlines to a minimum, there shouldn't be any new lines between list items, only a maximum of one new line between numbered items, and 3 consecutive new lines or more should be merged into one.\n\n"
            "Discord-formatted text:\n{text}\n\n"
            "Telegram-formatted text:"
        )
        return PromptTemplate.from_template(template)


    @staticmethod
    def get_content_tags_prompt() -> PromptTemplate:
        template = """
        You are given the following content (executive summary + full summary) from a {media_type}:

        {full_text}

        Below is a list of possible tags. Return only the tags (exact text from the list) that are most relevant, separated by commas. 
        If none apply, return empty.

        Possible tags:
        {possible_tags}

        Answer with the chosen tags on a single line, comma-separated (no extra words).
        """
        return PromptTemplate.from_template(template)
