# Hotel Image Classification(HIC)

## Identification of attributes influencing customer evaluation of luxury hotel brands using hotel images 
Table of Contents :bookmark_tabs:
=================
- [Authors](#authors)
- [Abstract](#abstract)
- [Steps to Use the Application](#steps-to-Use-the-application)
- [Tech Stack Used](#tech-stack-used)

## Authors
###  Submitted in fulfilment of the requirements for award of degree Bachelor of Technology in Computer Science and Engineering

<tr>
  <td>By</td>
  <td>Ali Asgar Saifee (18JE0077)</td>
  </br>
  <td>Chirag Jain (18JE0254)</td>
  </br>
  <td>Sanket Agarwal (18JE0730)</td>
  </br>
</tr>

### Under the guidance of

<tr>
  <td>Prof. Rajendra Pamula</td>
  </br>
  <td>Assistant Professor</td>
  </br>
  <td>(Dept. of Computer Science and Engineering)</td>
 </tr>


## Abstract

The objective of the research is to analyse luxury hotel pictures posted online in social
media by the tourists to understand the different hotel attributes (bedroom, living
room, dinner room, restaurant etc.,) influencing the consumer evaluation of luxury hotel
brands.
</br>
</br>
Users seek out hotel photos before making a decision to book. The research evaluates
consumers’ visual data on TripAdvisor through a deep learning approach. Four different
deep CNN architectures were trained and compared, of which EfficientNet B4
performed the best, hence was used to classify the images in actual dataset.
</br>
</br>
Results shed light on the significant part of non-textual elements of the hotel
experience such as pictures, which cannot be explored through traditional methods. In
particular, the analysis of 9,251 consumers’ pictures leads to the identification of the
attributes that had the higher impact on their experience. These attributes emerged as
specific features of interior elements of the hotels (restaurant, bedroom and bathroom).
</br>
</br>
Finally, the study shows how deep learning algorithms can -
</br>
● help monitoring social media and understand consumers perception of luxury
hotels through the new analysis of visual data, and
</br>
● turn into better brand management strategies for luxury hotel managers

## Steps to Use the Application

- Clone the repository onto your own local machine
- Download models folder from [here](https://drive.google.com/drive/folders/16gc0LcNiMgtSSS2wLwtc1qFLHwkEur9z?usp=sharing) as .zip and extract it inside the repository. 
- Open command prompt/terminal
- Run pip install -r requirements.txt
- Type ‘streamlit run app.py’ in the command prompt/terminal
- A localhost address should automatically open in your web browser. If not, copy the local URL from the command prompt/terminal into your web browser.
- Click 'Browse files' and upload an image file in .jpg format.
- It would show the prediction of HIC model.

## Tech Stack Used
- Python: Version 3.7.4
- Packages: PIL, torchvision, torch, streamlit, albumentations, efficientnet_pytorch