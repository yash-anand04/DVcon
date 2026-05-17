



DVCon India 2026: Design Contest
Stage 2A Q&A Session


## Recording Link:
https://marvell.zoom.us/rec/share/9nhB7LYzpAZ1VnOnJ3aYGM4oMzyX8hglHSnhDBgrTNZeDG
3xO8bxb_8rZHlYVj3j.RDT27TSnC7OZkAkq
Passcode: 67m7GMA.

Important dates:
Stage 2A result and report submission : May 5
th
## 2026
Stage 2A Result announcement : May 20
th
## 2026

Stage 2A Deliverables:
- Code for functionally correct for object detection based on input queries
- A two-page report: One page on the approach description and one page on results
snapshot
- Video demonstration on the application

Meeting summary
Quick recap
The meeting focused on Stage 2A of a challenge competition, where Professor Vivek and the
team provided guidance on implementing object detection applications. The main requirement
was for teams to develop a functional pipeline that can process images and respond to 14
specific questions, with emphasis on correct functionality rather than optimization or speed.
Teams were instructed to focus on CPU-based software implementation using any programming
language, with hardware acceleration being optional. The deliverables include source code, a
two-page report describing the approach and results, and a video demonstrating the working
application. The submission deadline was set for May 5th, though a small extension may be
considered due to some teams having exams. Teams were also informed that while they can use
additional datasets beyond the provided COCO dataset, they must still be able to answer the 14
specified questions to meet the evaluation criteria.
Next steps


- All teams: Develop and submit a functionally correct application pipeline (source code)
that processes input images and queries, demonstrating object detection and reasoning
as per the 14 provided queries; code should run on CPU (not GPU) for inference.
- All teams: Submit a two-page report: one page describing the approach and any
modifications from the original proposal, and one page with snapshots of results.
- All teams: Submit a short video demonstrating the working of the application.
- All teams: Submit all deliverables (code, two-page report, video, and optional hardware
implementation details if applicable) by May 5th (with possible slight extension to be
confirmed).
- Seru: Send a detailed email via EasyChair to all teams specifying deliverables,
submission format, and updated deadline.
- All teams: If team composition changes are needed, send an email to Seru with all
current and new team members in copy, specifying members to be dropped/added.
- Organizers (Seru/Vivek/Upul/Bibhas): Internally discuss and confirm if a couple of days'
extension to the May 5th deadline is possible, and communicate the final deadline to
teams.
- Organizers: Share the recording of this meeting and the detailed deliverables email via
EasyChair.
- Organizers: Share Vega IP and conduct a workshop on accelerator IP integration after
Stage 2A submission/evaluation.
## Summary
Stage 2A Application Pipeline Development
Professor Vivek outlined the focus for Stage 2A of the challenge, emphasizing the need to
develop and test the proposed application pipeline using the provided dataset and 14 questions
for evaluation. The main goal is to ensure the application runs functionally correctly, with the
possibility of later accelerating it using free tools like Vivado. While the specific deadline was not
discussed in detail, the team was advised to focus on demonstrating a working application in this
stage.
Stage 2A Object Detection Application
The team discussed the focus for Stage 2A, which is ensuring the application is functionally
correct for object detection based on input queries. Vivek emphasized that the application should
run properly on a CPU for inference, though GPUs can be used for training, and suggested
exploring FPGA or accelerator designs as additional information. The team was instructed to
submit their source code, a report with result snapshots, and a short video demonstrating the
application functionality.
Stage 2A Submission Requirements Discussion
The team discussed submission requirements and deadlines for Stage 2A of the project. Seru
outlined the deliverables, which include code, a report, a video of the application working, and a
snapshot of final results. The submission deadline was set for May 5th, with results to be
announced by May 20th. Vivek assured participants that they could use any available FPGA
board for hardware implementation and encouraged submissions regardless of the application's
completeness, as the team would evaluate all progress made.


Deliverable Requirements and Evaluation Criteria
The team discussed deliverable requirements, with Professor Vivek confirming that students
need to submit a two-page report including one page on their approach description and one page
on results snapshot, rather than using the previous template. Ankush raised concerns about
accessing the Genesis 2 board in Vivardo, but Professor Vivek advised him to proceed with any
available FPGA board and focus on running the application and designing acceleration blocks.
The discussion clarified that the model's evaluation will be based on 14 specific questions related
to object detection accuracy, with emphasis on appropriate object classification rather than just
basic detection.
## Application Development Process Discussion
The team discussed the application development process, with Vivek clarifying that Vivardo is
not mandatory at this stage and the focus is on getting the source code running in Python. Dr.
explained the pipeline should consist of two stages: first testing in software environment for
object detection and scene understanding, then implementing in hardware. HP raised concerns
about dataset download issues and accuracy, specifically regarding fire extinguishing and potato
tasks having insufficient images for training. Rohith addressed a question about Vega IP core
communication, clarifying it would be handled in stage 3 and not a current concern.
## Second Stage Deliverables Discussion
The team discussed deliverables for the second stage, which focuses on creating a working
pipeline. Vivek clarified that while hardware implementation is optional, the main requirement is a
software-based working pipeline. The team also addressed questions about using the given
COCO dataset, with Vivek confirming it can be extended with other datasets but is not
mandatory. Additional clarifications were provided regarding simulation requirements, with Vivek
confirming that both software and hardware (RTL) implementations can be submitted if
completed, though the software version is mandatory for this stage.
## Detection Model Implementation Requirements
The team discussed implementation requirements for a detection model, with Dr. explaining that
participants should aim for accuracy as close as possible to the paper's implementation, focusing
on 14 specific tasks as a minimum requirement. The group clarified that software implementation
can use any programming language including Python, Jupyter Notebook, or Google Colab, with a
maximum team size of 3 members. While some students requested deadline extensions due to
exams, Professor Vivek indicated that only a couple of days extension might be possible, with
May 5th remaining the current submission date.
