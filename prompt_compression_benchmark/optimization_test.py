# Requires the following branch of LLMLingua: https://github.com/cornzz/LLMLingua/tree/timings

from llmlingua.prompt_compressor import PromptCompressor
import tiktoken
import time
import nvtx

model = "llmlingua2"
device = "cuda:4"
batch_size = 50
models = {
    "llmlingua2": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    "llmlingua2_small": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
}
tokenizer = tiktoken.encoding_for_model("gpt-3.5")
print("Loading model...")
compressor = PromptCompressor(
    model_name=models[model],
    device_map=device,
    use_llmlingua2=True,
    llmlingua2_config={"max_batch_size": batch_size},
)
print("Model loaded\n")

prompt = "The report of the Civil Rights, Utilities, Economic Development and Arts Committee Agenda Item three Resolution 31669 Encouraging as a best practice the use of an individualized tenant assessment using the Fair Housing Act's discriminatory effect standards to avoid Fair Housing Act violations when criminal history is used as a screening criterion in the Landlord Screening Process, Committee recommends that the resolution be adopted as amended grade. I move to amend Resolution 31669 by substituting D four for version D three, which includes a new attachment. A And I understand Councilmember Bagshaw also has an amendment, but let's first, if we could, let me just go through the changes to the resolution since the last committee meeting. The changes are found in two recitals, as well as sections one and five are the primary changes. We added a recital that again lifts up the HUD guidance to show that a criminal history screening policy is next must serve a substantial, legitimate and nondiscriminatory interest. Another recital we included was referencing the the Seattle Fair Chance Employment Ordinance and the way they approach some of these same issues, looking at making sure that individualized assessments and prohibiting questions on initial job applications regarding an applicant's criminal record. And then in Section one, and these are changes that we worked on with stakeholders together with councilmembers, the wants office desiring to really focus on not the the impacts of this particular resolution, but for what we hope to see in the future ordinance that is going to be coming to us to to regulate this area of of housing screening practices. And it identifies the principles that came out of the Hallow recommendations. And in Section five, again, this is just clarifying that the expectation in the HUD guidance is to distinguish between criminal conduct, that that indicates a demonstrable risk to residents safety and conduct that does not the resolution itself, whereas it's really focused on encouraging landlords to to follow HUD guidance that has been recently released regarding criminal records. The separate sections do. A couple of different things. Sections one and two, again, focus specifically on the future legislation that we expect to be coming out of the mayors task force. The. The next section basically says that we endorse practices that landlords should not automatically exclude individuals for housing on the basis of prior event arrests. The further sections refer refer to the process that the Office of Housing has facilitated to create procedures to select tenant screening agency guidelines for property management and affordable housing. Another section recommends that the that a landlord should not rely on records that cannot be reported by consumer reporting agencies under state law. And another section focuses on the Office of Civil Rights efforts to basically do enforcement of existing fair housing laws through testing, investigation of charges and other other means. The final section requests that as OCR when determining whether or not a complaint of housing discrimination based on the use of criminal history, whether or not there should they should it ask them to seek to determine whether or not there's a disparate impact? So that's an overview of both the resolution and the the changes that have been made since the committee discussion and vote on June 3rd. And I don't I may have started talking before I had a second. May I add a second? All right, great. Those in favor of supporting the substitute version D for 4d3ii in a OC. So we have the substitute amendment before us. Councilmember Bagshaw will move to further amend the resolution, but before consideration of the amendment, we have to move to suspend the rules because we've got we received the text for the amendment after the , I believe, noon deadline today. So I moved to suspend Council Rule three A6 relating to presentation of full council amendments before 12 noon, checking all those in favor of the motion carries and we can proceed with consideration of the proposed amendment. Great. Thank you very much. What I am proposing is the addition in our whereas section two recognize what we all worked on a year ago called the Certificate of Restoration of Opportunity or the acronym was CROP. And it really and it's designed to address what the gentleman in the front row had talked about earlier today during public testimony , the state legislature passed unanimously out of both houses of the Act around Certificate of Restoration of opportunity. And what it is designed to do is to offer potential public and private employers or housing providers concrete and objective information about an individual who has served his or her time in prison and is now been released. And what we're really wanting to do here is to reintegrate individuals who have had a previous criminal history to provide housing and employment opportunities so that whereas that I am recommending we ensure it comes right out of the bill. And it would say that in an act relating to certificates of restoration of opportunity that will offer potential public and private employers or housing providers concrete and objective information about an individual under consideration for an opportunity. These certificates can facilitate the successful societal reintegration of individuals with a criminal history whose behavior demonstrates that they are taking responsibility for their past criminal conduct, pursuing a positive, law abiding future. So I'm just suggesting we add this, that it refers to this legislation, which I will hope a court will provide certificate, a restoration of opportunity, and an individual has something else in his or her hand to help him get a job or housing. So we have a first. We moved it and we second it as well. No. Okay. We have a move in a second. Now, all those in favor of the amendment to the resolution say I, I, I opposed me. And now we will have the full version before us to vote on any comments. Comment. Sorry, I have just some closing statements. I just really I think it's so important that landlords, housing providers in this community understand what the law is when it comes to developing policies and practices for making decisions based on criminal history. We know that we're not likely to have the ordinance that will do this work and until after the Mayors for Fair Chance Housing Committee will be reconvened in July, and they will have a series of meetings before they bring to us recommendations for for an ordinance. And so in the interim, it's really important that we lift up the the policies that HUD is currently currently promulgating and making sure that both landlords are engaged with the policy direction that the that the city is going to be pursuing in the future, as well as protecting themselves from fair housing complaints today. So with that those in favor of adopting the resolution vote i. I. Those oppose vote no. The motion carries the resolution is adopted and the chair will sign it. And now we can read items for through eight together."
prompt *= 16
print("GPT-3.5 tokenizer len:", len(tokenizer.encode(prompt)))
prompt_len = len(compressor.tokenizer.encode(prompt))
print("Prompt len:", prompt_len, "number of sequences:", prompt_len / 512)
repetitions = 5

times_total, times_model = [], []
for i in range(repetitions):
    print("-- Benchmark Repetition", i)
    comp_rng = nvtx.start_range(message="compress_prompt_llmlingua2", color="purple", domain="functions")
    compressed = compressor.compress_prompt_llmlingua2(prompt, rate=0.5, return_timings=True)
    nvtx.end_range(comp_rng)
    times_total.append(compressed["timings"]["total"])
    times_model.append(compressed["timings"]["model"])
    print("Total:", compressed["timings"]["total"], "- Model:", compressed["timings"]["model"])
    
    # time.sleep(10)

# TODO: TEST BATCH=100 AND LARGE PROMPT

# print("\nPrompt:\n\n", prompt, "\n---\nCompressed:\n\n", compressed["compressed_prompt"])
print(f'\n\nAVERAGE:\nTotal: {sum(times_total) / repetitions} - Model: {sum(times_model) / repetitions}')