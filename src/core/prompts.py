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
        else:
            # Article
            summary_template = """
            You are an analytics professional at Lido Finance ... (Article summary)
            {text}
            
            YOUR NOTES:
            """
            refine_template = """
            You are an analytics professional at Lido, a Liquid Staking protocol for Ethereum. You are given a transcript of Twitter Spaces in the crypto/web3 space that may or may not be related to Lido.
            Given the transcript, you are refining structured notes in markdown format. Think of your notes as key takeaways, TLDRs, and executive summaries.

            Here is the existing note:

            {existing_answer}

            New context:
            {text}

            Refine ...
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
        else:
            # Default: Article
            exec_template = """
            Given the summary of an Article:
            {text}

            Generate an extremely brief executive summary for Lido contributors. It should be concise, focused, and only contain information relevant to Lido Finance. 
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