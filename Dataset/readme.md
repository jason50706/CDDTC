We describe the meaning of each field in the data here.

- id: The number before _ is the fact check report id specify by TFC, and the nuber j after _ means this rumor is the jth rumors related to this fact check report. (e.g. 5702_2 means this rumors is the second rumor related to fact check report 5702)
- original_text: The rumors itself, which is the input of model.
- title: The title of the fact check report.
- verdict: The verdict made by TFC according to their fact check to this rumor.
- topic: The topic TFC think this rumor related to.
- url: The url to this fact check report.
- publish_date: The date TFC publish this report on their website.
- char_labels: The label of each character in original_text. B means this character is the beginning of a claim, I means that character is inside a claim span, O means that character isn't a begining or inside of any claims. (e.g. If the first character in char_labels is B, meaning the first character in original_text is a beginning of a claim.)
- span_labels: Specify each spans' start and end position in the original_text, each list represent one claim span.

If you approach claim span detection using sequence labeling model, you can use char_labels as label; If you approach it as a start/end position identification model, you can use span_labels as label.