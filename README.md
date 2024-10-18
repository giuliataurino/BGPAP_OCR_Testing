# OCR_Testing

## Overview

This repository contains a series of pipelines for the testing and evaluation of pre-trained OCR models on digitized archival photographs. The tests were run on a subset of images from the Boston Globe Photo-Archive and they focused on the verso of digitized photographs, which included handwriting, typed text and stamped text. This repository is part of the Boston Globe Photo-Morgue Project, funded by the NEH Digital Humanities Advancement Grant under the broader project "Machine Learning for Large-Scale Journalism Collections" [[HAA-296412-24](https://apps.neh.gov/publicquery/AwardDetail.aspx?gn=HAA-296412-24)]. 

## Team

**Main Team**

Giulia Taurino, Ph.D., Khoury College of Computer Sciences (Northeastern University)

Prof. David Smith, Khoury College of Computer Sciences (Northeastern University)

Sarah Sweeney, Head of Digital Production Services, NUASC (Northeastern University Library)

**External Project Collaborators**

Supervisor: Prof. Philip Bogden, Roux Institute (Northeastern University)

Students: Seamus Mawe, Roux Institute (Northeastern University); Daniel O'Brien, Roux Institute (Northeastern University); Thomas Henehan, Roux Institute (Northeastern University)

## Dataset

The Northeastern University Archives and Special Collections (NUASC) holds an archive of over 5 million photographs from the Boston Globe past issues and news cycles. The scanned records include published and unpublished photographs (recto and verso), negative photographs and newspaper clippings. The printed photographs contain handwriting, typed text and stamped text in the back of the image.

## Goals
The Boston Globe Photo-Morgue Project sets out to create a ML workflow for archivists and librarians to automatically extract handwritten and typewritten text from the scanned photographs. The goal is to make this data, including information on photographer's name, date, location, subjects and other labels, machine-readable and usable for metadata enrichment. For this purpose, a team at Northeastern University tested and evaluated pretrained, open source OCR models. The tasks required the following phases: 
1. Manually annotate 150 images in Transkribus for supervised learning and/or testing of pretrained, open-source models; 
2. Test and evaluation of the following pretrained, open-source OCR models: Kraken, Tesseract, PaddleOCR, EasyOCR, DocTR.

## Phase I

### Annotations with Transkribus

Tranksribus is an [online](https://readcoop.eu/transkribus/), AI-powered platform for digitization, text recognition and transcription, and searching of historical documents. It was created by a group of European research institutions as part of the Horizon 2020 "READ" EU project and is maintained by the READ CO-OP, an LLC created in 2019 for this purpose.

The main team selected a random sample of images from the Boston Globe Photo-Morgue, which were converted to JPGs to enable upload to Transkribus.

The images were annotated in Transkribus using the following steps:

1. Draw a bounding box around text.
    * The bounding box can be modified with the selection tool to better cover rotated text
2. Draw a base line below each line of text 
    * The base line sould be drawn in the English reading direction, left to right (e.g. right to left for upside-down text)
3. Transcribe the text in the transcription field.
    * This can include looking at the fronts of the images, as well as looking up names of boston landmarks to interpret some text.
    * Characters should be transcribed directly preserving misspellings, abreviations, etc. (e.g. in one image "night" was spelled "nite" and was transcribed as such)  

| Image | Bounding Box | Lines | Transcription|
|:---:|:---:|:---:|:---:|
|<img src="docs/img/blank.png" title="Text" alt="blank" height="200"/> | <img src="docs/img/box.png" title="Bounding Box" alt="blank" height="200"/> |<img src="docs/img/lines.png" title="Lines" alt="blank" height="200"/> | <img src="docs/img/transcription.png" title="Transcription" alt="blank" height="200"/> |

The manual transcriptions in Transkribus will be used as the targets for our different supervised machine learning methods. Basically, the annotations will be used to assess the quality of the OCR models. Transkribus's OCR model will be used in the next phases of the project to transcribe the text on the backs of the photos.


## Phase II

### Testing and Evaluation





