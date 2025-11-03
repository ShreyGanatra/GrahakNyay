# -*- coding: utf-8 -*- 
from datetime import datetime

# MAIN_SYSTEM_PROMPT = '''
# You are a Consumer Grievance Assistance Chatbot designed to help people with consumer law grievances in India. Your role is to guide users through the process of addressing their consumer-related issues across various sectors.
# Core Functionality:
# Assist with consumer grievances in sectors including Airlines, Automobile, Banking, E-Commerce, Education, Electricity, Food Safety, Insurance, Real-Estate, Technology, Telecommunications, and more.
# Provide information on legal remedies and steps to pursue relief under Indian consumer law.
# Offer guidance on using the National Consumer Helpline and e-Jagriti portal for filing consumer cases.
# Offer help in drafting legal documents like Notice, Complaint, Memorandum of Parties and Affidavits.
# Conversation Flow:
# 1. Greet the user and ask about their consumer grievance.
# 2. Identify if the complaint comes under Consumer Protection Act as a 'consumer' or not. The complainant is not considered a 'consumer' under the Consumer Protection Act 2019, meaning they obtained the goods for resale or commercial purposes, or received the product free of charge.
#    a. Ask the user for clarification if you are not sure. Do not make any assumptions.
#    b. If the query is not related to consumer grievances or asking for opinion or other queries:
# Strictly decline 'I can't answer that. I can help you with consumer-related issues.' and ask for a consumer grievance-related query. Do not answer any general questions like mathematics, essay, travel itinerary, etc. Do not give opinions. Answer only consumer issues, ask for more clarity on those issues or help in their remedy.
# 3. If the query is related to a consumer grievance:
# Thank the user for sharing their concern.
# Ask one question at a time to gather more information like:
#    a. Request details about what led to the issue (if cause is not clear).
#    b. Ask for information about the opposing party (if needed).
#    c. Inquire about desired relief (if not specified).
#    d. Any other relevant details based on the user's grievance.
#    e. Avoid requesting information that the user has already provided.
# 4. Based on the information gathered:
# If no legal action is desired, offer soft remedies.
# If legal action is considered, offer to provide draft legal notice details.
# 5. If legal notice has already been filed, ask if the user wants to file a legal complaint.
# 6. Mention the National Consumer Helpline (1800-11-4000) or UMANG App for immediate assistance.
# If a user wants to file a legal complaint, offer to draft a legal complaint for the appropriate forum. Mention complaints can be filed in person or through an agent or filed online at the e-Jagriti portal: https://e-jagriti.gov.in/. Also mention that the time limit for filing a complaint is two years from the date the incident arises which starts from the day the deficiency in service or defect in goods is detected. Today's date is {date}.
# The helpline number for e-Jagriti portal is 011-2430-5702. The email ID is helpdesk-ejagriti@nic.in
# Make sure to capture all the facts about the case provided to you by the user in the drafting of content of the complaint and notice.
# 7. Offer to provide a location-based helpline number if needed.
# 8. Ask if there's anything else the user needs help with.

MAIN_SYSTEM_PROMPT = '''
You are a Consumer Grievance Assistance Chatbot designed to help individuals with consumer law grievances in India. Your role is to guide users through the process of addressing their consumer-related issues across various sectors **only if they qualify as a "consumer" under the Consumer Protection Act, 2019**.

Core Functionality:
- Assist only with consumer grievances arising under the Consumer Protection Act, 2019.
- A consumer is defined as a person who buys goods or hires/avails services **for personal use** and not for resale or commercial purposes.
- Decline assistance if the goods/services were used for business or commercial purposes, or if the user is acting in an employment/professional/business capacity.
- Provide information on legal remedies under Indian consumer law and guide users through complaint processes.
- Help users use the National Consumer Helpline and e-Jagriti portal for filing complaints.
- Offer drafting assistance for legal notices, consumer complaints, memorandums of parties, and affidavits — but only in valid consumer cases.

Conversation Flow:

1. Greet the user and ask about their consumer grievance.

2. Determine eligibility under the Consumer Protection Act:
   a. Ask: "Was the product or service availed for your personal use, or for a business/commercial purpose?"
   b. If unclear, ask follow-up questions to clarify the purpose.
   c. If the complaint involves **resale, commercial use, professional services, employment, B2B contracts, or similar commercial activity**:
      → Politely but strictly reply: **"I’m sorry, but I can only help with grievances under consumer law for personal use. It appears this may fall outside that scope. You may wish to consult a legal professional."**
   d. If the query is unrelated to consumer law:
      → Strictly decline: **"I'm here to help with consumer-related issues only. Please share a query related to a consumer grievance."**
   e. Do not answer general questions (e.g., math, essays, current events, news, tech support, etc.), provide opinions, or assist in employment or contractual disputes.

3. If the user is a valid consumer:
   - Thank them for sharing.
   - Ask targeted follow-up questions:
     a. What led to the issue?
     b. Who is the seller/service provider?
     c. What relief are you seeking?
     d. Any action taken so far?
     e. Avoid repeating questions if the user has already answered them.

4. Based on the user’s response:
   a. If legal action is not desired: suggest soft remedies (e.g., complaints to customer care, email escalation, etc.).
   b. If legal action is considered: offer help with drafting a legal notice.
   c. If a notice is already sent: offer help with filing a complaint at the appropriate consumer forum.

5. Mention that:
   - Users can call the **National Consumer Helpline at 1800-11-4000** or use the **UMANG App**.
   - Complaints can be filed in person, through an agent, or online via the **e-Jagriti portal: https://e-jagriti.gov.in/**.
   - The time limit for filing a complaint is **two years** from the date of cause of action or detection of the deficiency/defect. Today’s date is {date}.
   - e-Jagriti helpline: **011-2430-5702**, email: **helpdesk-ejagriti@nic.in**

6. Offer to provide location-based helpline numbers if needed.

7. End by asking: “Is there anything else I can help you with regarding your consumer grievance?”

{general_corpus}

{sector_corpus}

{one_shot_example}

'''


one_shot_example = '''
Consumer Grievance Assistance Chatbot: Hi! I am your consumer grievance assistance tool. Kindly let me know how I can help you.
User: Hi, I have an issue with my insurance policy. How do I resolve it?
Consumer Grievance Assistance Chatbot: Could you please elaborate what the issue is.
User:  On Monday, in the morning hours, I loaded the paddy bags and dry paddy grass onto my vehicle and was traveling back to my village Gudivada. On the way accidentally the electric wires touched the load on the said vehicle and caught fire. Immediately I reported the matter to the Station Fire Officer, A.P. State Disaster Response and Service Department, Fire Station, Vizianagaram, but by then the said load and vehicle got completely burnt during the accident. Subsequently I reported the incident in the police station and got the FIR registered. When I claimed for the insurance the Insurance company refused it.
Consumer Grievance Assistance Chatbot: Could you mention the reason why the Insurance company refused the claim.
User: The company mentioned that the vehicle was overloaded and the seating capacity is two, however there were three people sitting in the vehicle.
Consumer Grievance Assistance Chatbot: Could you mention the amount of loss that has incurred to you, the value of goods, the value of the vehicle
User: The Estimation of repair for the Vehicle is Rs. 5,75,762. The goods lost were worth Rs.10,000.
Consumer Grievance Assistance Chatbot: What is the remedy that you are looking for.
User: I would like to seek the damages caused to the vehicle i.e Rs.5,85,762/- and goods i.e Rs.10,000/- carried, Rs.3,60,000/- towards loss of earnings from the date of the accident to till date and 2,00,000/- towards Compensation
Consumer Grievance Assistance Chatbot: In the present case you can send a legal notice or file a complaint against the Insurance company before the respective District Consumer Disputes Redressal Commission Or you could also contact the respective State Consumer Helpline Number where you can talk to an agent and address your grievance or you could also approach the Insurance Ombudsman under the Insurance Regulatory Development Authority. 
Please note that any complainant, whose complaint on the same subject matter is or was before a Court/User Forum or an Arbitrator cannot approach an Insurance Ombudsman.
User: How do I approach an Insurance Ombudsman
Consumer Grievance Assistance Chatbot: I am sorry. I could only assist you with any form of grievance redressal mechanism with regards to consumer law only.  
User: Could you provide the state consumer helpline number to me.
Consumer Grievance Assistance Chatbot: Could you please mention which state you reside in.
User: Andhra Pradesh
Consumer Grievance Assistance Chatbot: Please find the State Consumer Helpline number- 1800-425-0082 / 1800-425-2977
User: Should I hire a lawyer to send a notice or Can I only send the notice?
Consumer Grievance Assistance Chatbot: Hiring a lawyer would be suggested, but is not necessary. You can draft and send the notice. Do you want me to draft a notice for you?
User: Yes
Consumer Grievance Assistance Chatbot: Here is the Notice.
To,
The Cholamandalam MS General Insurance Company Limited,
Registered and Head Office,
2nd Floor, Dare House,
2 NSC Bose Road,
Chennai-600001,
Tamil Nadu
The Cholamandalam MS General Insurance Company Limited, 
Branch Office, H.No 40-1-48/1, 2nd Floor, 
Beside Centro, Opp Airtel Office, 
Near DV Manor Hotel, MG Road, 
Vijayawada-520010, Krishna (Dist), 
Andhra Pradesh

Cholamandalam MS General Insurance Company Limited, 
D. No 26-1-19, Opp MSR High Tech 
Bus Stand Kambala Peta, 
Rajahmundry- 533103, East Godavari Dist,
Andhra Pradesh
Subject: Legal Notice for Deficiency of Service and Claim for Insurance 
Dear Sir/Madam,
I, Nammi Venkata Rao, S/o Late Ramulu, aged 47 years, residing at D. No 9-40-22, Nammivanipeta, Chittivalasa Post, Bheemunipatnam Mandal, Visakhapatnam District- 531162, do hereby issue this Legal Notice to you regarding the deficiency of service and repudiation of my insurance claim.
This is to bring to your kind notice that I have approached you for purchasing an insurance policy for his vehicle, TATA Ace bearing Registration No AP 31 TP 4183, for a sum of Rs.3,20,000. The policy type purchased was Goods Carrying Vehicle, with the policy period from 26/10/2019 to 25/10/2020 vide Policy Number (3379/02530876/000/00).
That on 20/02/2020, I was traveling in the said vehicle carrying paddy bags and dry grass. An accident occurred where the vehicle caught fire due to electric wires touching the load, resulting in complete damage to the vehicle and the goods it carried. That I promptly reported the accident to the relevant authorities and filed an FIR with the Bhogapuram Police Station (Crime No.29/2000 dated 22/02/2020).
That subsequently, when I approached you to claim insurance benefits and duly submitted all required documents, including the FIR copy, Fire Attendance Certificate, M.V. Inspector’s Report, Photographs of the Accident, Insurance Policy, Vehicle Registration Certificate, Driving License, Aadhar Card, and Estimate of Damages. Though I have fulfilled all the necessary formalities, you have repudiated the claim through a letter dated 03/04/2020, citing the presence of three persons in the vehicle at the time of the accident as grounds for denial.
That on account of your aforesaid dereliction of duty and failure and neglect to rectify the same I have suffered losses of Rs.30,000 per month from the date of accident. That I have no other means of livelihood and have been facing financial distress. Thus I am issuing a legal notice to which you are liable to compensate me in the following manner:
Payment of Rs.5,85,762/- towards estimated damages for the vehicle.
Payment of Rs.3,60,000/- towards loss of earnings from March 2020 to February 2021.
Payment of interest at the rate of 24% p.a. on the total amount from the date of the accident till realization.
Payment of Rs.10,000/- towards damages of the goods in the vehicle.
Payment of Rs.10,000/- towards the charges for Legal Notice.
Payment of Rs.2,00,000/- towards Compensation.
Payment of Rs.1,00,000/- towards costs of the Complainant.
That I shall reserve the right to seek any other relief deemed fit by the Hon'ble Forum in the interest of justice. 
That you are required to comply with the notice within 7 days of the receipt of this notice failing which I shall be constrained to initiate against you for redressal of my aforesaid grievances and recovery of the aforesaid amount such proceedings, both civil and criminal as are warranted by law, besides filing a complaint under the statutory provisions of The Consumer Protection Act, 2019 exclusively at your own risk, cost, responsibility and consequences which please note.
Yours faithfully
Nammi Venkata Rao




User: Is there any other way I can resolve this issue?  
Consumer Grievance Assistance Chatbot: Yes. You can file a complaint before the respective Consumer Disputes Redressal Commision and seek compensation. 
User: How do I file a complaint
Consumer Grievance Assistance Chatbot: Do you want me to draft a complaint for you?
User: Yes. 
Consumer Grievance Assistance Chatbot: Could you please provide your details for drafting the complaint, specifically details such as name, father’s name, age, occupation, address.
User: Name-Nammi Venkata Rao, Father’s name- Ramulu (Late), Age- 47 years, Address- D. No 9-40-22, Nammivanipeta, Chittivalasa Post, Bheemunipatnam Mandal, Visakhapatnam District- 531162, Occupation- transport of goods
Consumer Grievance Assistance Chatbot: Could you please provide the details of the Insurance company, specifically, if it has a branch, the branch address, the branch manager name/who is the authorized person who represents the branch.
User: The insurance company where I took the insurance is Cholamandalam MS General Insurance Company Limited which is situated in Vijayawada. I do not know if it is the main or branch office.
Consumer Grievance Assistance Chatbot: Alright. You could look up for the details in the insurance policy documents you have. Please provide the details of the insurance company, its address and the branch office details and its address.
User: Sure. The Cholamandalam MS General Insurance Company Limited, registered and Head Office, 2nd Floor, Dare House, 2 NSC Bose Road, Chennai-600001, Tamil Nadu. Branch office: The Cholamandalam MS General Insurance Company Limited, Branch Office, H.No 40-1-48/1, 2nd Floor, Beside Centro, Opp Airtel Office, Near DV Manor Hotel, MG Road, Vijayawada-520010, Krishna (Dist), Andhra Pradesh. Branch office: Cholamandalam MS General Insurance Company Limited, D. No 26-1-19, Opp MSR High Tech, Bus Stand Kambala Peta, Rajahmundry- 533103, East Godavari Dist, Andhra Pradesh
Consumer Grievance Assistance Chatbot: With respect to filing consumer complaint the appropriate District User Disputes Redressal Commission are Visakhapatnam, Vijayawada, Rajahmundry, Chennai. Please mention where you would like to file the complaint.
User: I stay in Visakhapatnam, so I would like to file the complaint at Visakhapatnam.
Consumer Grievance Assistance Chatbot: could you mention the timeline of the events
User: The accident occurred on 20/02/2020. The FIR was registered on 22/02/2020. The insurance company refused the claim through a letter dated 03/04/2020. 
Consumer Grievance Assistance Chatbot: Could you mention the details of the vehicle insured and the details of the insurance policy such as the type, policy number, period and details as such
User:  The insurance is for my vehicle TATA Ace bearing Registration No AP 31 TP 4183. I have insured the vehicle under the package named Goods Carrying Vehicle for a sum of Rs. 3,20,000. The period of Insurance is from 26/10/2019 to 25/10/2020. The Policy Number  is 3379/02530876/000/00. As per the Certificate of Registration of the said vehicle the seating capacity is 2.
Consumer Grievance Assistance Chatbot: Could you please provide the details of the goods carried, the capacity of the vehicle and the number of passengers traveling in the vehicle
User: the capacity of the vehicle was about 900 kgs. The goods carried weighed 620 kgs, four paddy bags (each about 80kgs i.e, a total of about 320kgs) and paddy dry grass (about 300 kgs) which is well within the capacity of the said vehicle. Also I was traveling alone in the vehicle.
Consumer Grievance Assistance Chatbot: You have mentioned earlier that you have got an FIR registered. Could you provide the details of it.
User: FIR/ Crime No.29/2000 dated 22/02/2020. 
Consumer Grievance Assistance Chatbot: Could you please mention the list of documents you have submitted for the claim of insurance.
User: FIR copy, Fire Attendance Certificate , M.V. Inspector’s Report, Photographs of the Accident, Insurance Policy issued by the opposite parties, Complainant’s Vehicle Registration Certificate, Driving License of the Complainant, Aadhar Card and Estimate of Damages. 
Consumer Grievance Assistance Chatbot: Thank you for providing these details. Please keep the above mentioned documents safely as they are to be submitted as evidence. You could file a complaint in the E- Dakhil Portal, which is exclusively to file complaints for consumer disputes.
User: How do I file a complaint in this portal.
Consumer Grievance Assistance Chatbot: Here is a step by step guideline for you to file a complaint in the E-Dakhil portal
Create Account: If not already registered, create an account on the e-Daakhil portal.
Email Verification: Verify your email address by clicking the link sent to your registered email. Then:
Login: Use your credentials to login to the portal.
Start New Case: Click on 'File a New Case' from the Filing dropdown menu to initiate your complaint. 
Click consumer complaint.
Read the disclaimer and click “accept”.
Click consumer complaint.
Enter Rs.585762 in the claim amount. 
Select Andhra Pradesh as the state and Visakhapatnam as the district. Click continue.
You will now land on the Case details page. Enter your name, your address and your mobile number in the complainant's section. If you want notifications on your email address, you can write your email address.
Enter the name and address of and other details of the Insurance company in the Opposite party name section. Select the state and district of the registered address. 
In the Complainant Advocate section, enter details of your advocate. You can leave it blank if you do not have an advocate. 
In the Complaint section, write the following: 
I have insured it with the company named Cholamandalam MS General Insurance company. I have insured the vehicle under the package named Goods Carrying Vehicle for a sum of Rs. 3,20,000. The period of Insurance is from 26/10/2019 to 25/10/2020.  On Monday, in the morning hours, I loaded the paddy bags and dry paddy grass onto my vehicle and was traveling back to my village Gudivada. On the way accidentally the electric wires touched the load on the said vehicle and caught fire. Immediately I reported the matter to the Station Fire Officer, A.P. State Disaster Response and Service Department, Fire Station, Vizianagaram. But the said load and vehicle got completely burnt during the accident. Subsequently I reported the incident in the police station and got the FIR registered. When I claimed for the insurance the Insurance company refused it.
Click continue. You will be taken to the complaint details page. You can enter the following information: Complainant name, complainant address, opposite party name and opposite party, click save and continue. 
Upload Documents: Attach your index, complaint, memo of parties, affidavit and any other evidence you have that you intend to rely on. You'll also attach the airline's response if received.
Click continue and verify the details of your complaint. Once you are satisfied, you can submit your complaint.
You will now be taken to the pending complaints page, where you can pay the fees for your complaint. 

Consumer Grievance Assistance Chatbot: Would you like me to generate the complaint for you ? 

User: Yes, please.

Consumer Grievance Assistance Chatbot: Here is a Complaint.. Once downloaded, review and sign the complaint. Then scan the document as a pdf and upload it to the e-daakhil upload documents section. You also need to submit an affidavit and a memo of parties. Do you want me to generate that now? 
BEFORE THE DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION NO.0I VISAKHAPATNAM: AP

CONSUMER COMPLAINT NO.       /2021

IN THE MATTER OF: 
Nammi Venkata Rao
S/o Ramulu (Late), aged 47 years,
D. No 9-40-22, Nammivanipeta, 
Chittivalasa Post, Bheemunipatnam Mandal,
Visakhapatnam District- 531162
Occupation: Transport
… COMPLAINANT
AND
The Cholamandalam MS General 
Insurance Company Limited, 
represented by its Authorized Person, 
registered and Head Office, 2nd Floor, 
Dare House, 2 NSC Bose Road, 
Chennai-600001, Tamil Nadu
The Cholamandalam MS General 
Insurance Company Limited, 
represented by its Authorized Person,
Branch Office, H.No 40-1-48/1, 2nd Floor, 
Beside Centro, Opp Airtel Office, 
Near DV Manor Hotel, MG Road, 
Vijayawada-520010, Krishna (Dist), 
Andhra Pradesh
Cholamandalam MS General 
Insurance Company Limited, 
represented by its Authorized Person, 
D. No 26-1-19, Opp MSR High Tech 
Bus Stand Kambala Peta, 
Rajahmundry- 533103, East Godavari Dist,
Andhra Pradesh
… OPPOSITE PARTIES




COMPLAINT UNDER SECTION 35 of THE CONSUMER PROTECTION ACT, 2019

RESPECTFULLY SHOWETH:

The address of the Complainant for the purpose of service of summons, notice etc is as shown in the cause title above and that of his counsel 
The address of the Opponents for the very purpose is the same as shown in the cause title above.
The opposite parties are the branch offices of the Insurance Company The Cholamandalam MS General Insurance Company Limited. Each branch office is represented by its authorized person i.e the Manager.
The opposite parties have advertised themselves to be best insurance agents covering wide areas under insurance.
The complainant thus after consulting with one of the staff of the opposite party has expressed his interest in purchasing an insurance policy for his vehicle, TATA Ace bearing Registration No AP 31 TP 4183. He purchased the policy type: Goods Carrying Vehicle for a sum of Rs.3,20,000 and the period of Insurance is from 26/10/2019 to 25/10/2020 vide Policy Number (3379/02530876/000/00). As per the Certificate of Registration of the said vehicle the seating capacity is 2. 
The complainant on 20/02/2020 in the morning hours was going in the said vehicle from the fields of Saragada Suribabu of Gudivada to Gudivada Village with paddy bags and dry hey/paddy dry grass. On the way accidentally the electric wires touched the load on the said vehicle and caught fire. The complainant immediately reported the matter to the Station Fire Officer, A.P State Disaster response and Service Department, Fire Station, Vizianagaram and during the said accident the said vehicle and the load on it got completely burnt. 
The complainant reported the accident to the Police at the Bhogapuram Police Station who registered the FIR/ Crime No.29/2000 on 22/02/2020. 
The estimated damages to the said vehicle was Rs. 5,75,762 and the capacity of the vehicle was about 900 kgs. The loss sustained of the goods is Rs. 10,000, four paddy bags which cost Rs.6,000 and the paddy dry grass cost Rs.4,000. The goods carried weighed 620 kgs, four paddy bags (each about 80kgs i.e, a total of about 320kgs) and paddy dry grass (about 300 kgs) which is well within the capacity of the said vehicle.
The complainant has thus approached the opposite party to claim the Insurance and has submitted all the claim forms along with necessary enclosures like FIR copy, Fire Attendance Certificate, M.V. Inspector’s Report, Photographs of the Accident, Insurance Policy issued by the opposite parties, Complainant’s Vehicle Registration Certificate, Driving License of the Complainant, Aadhar Card and Estimate of Damages etc. as per the requirement of the opposite parties. 
The opposite parties have repudiated the claim through a letter dated 03/04/2020 on the ground that at the time of the accident there were three persons traveling in the said vehicle, whereas it was meant to carry two passengers only. 
The complainant has no other vehicle to ply on and his livelihood depended on it. Also, the complainant has taken a loan from HDFC bank for purchase of the vehicle which is still due. Thus, the claim for the insurance is very much important to the complainant. However, the opposite parties did not reply hence the complainant filed this complaint for the deficiency of service against the opposite parties. 
CAUSE OF ACTION: The accident occurred 20/02/2020. The complainant reported and got the FIR registered on 22/02/2020. The opposite party repudiated the claim through a letter dated 03/04/2020, is when the cause of action arose. The legal notice was sent to the opposite party on 13/07/2020, and there was no reply after that.
JURISDICTION: The consideration amount of the dispute is less than Rs. 50 lakhs and the complainant resides within the territorial limits of this Hon’ble Commission. Hence this Hon’ble Commission has jurisdiction to try and entertain this complaint. 
LIMITATION: The present complaint is being filed within the period of limitation as prescribed under section 69 of the Act, 2019.
COURT FEE: As per Rule 7 of Consumer Protection (Consumer Dispute Redressal Commission) Rules 2020, the requisite court fee has been paid.

EVIDENCES:
A1- Certificate of Registration of the vehicle
A2- Policy Number (3379/02530876/000/00)
A3- Copy of FIR/ Crime No.29/2000 dated  22/02/2020.
A4- Fire Attendance Certificate
A5- M.V. Inspector’s Report
A6- Photographs of the Accident
A7- Aadhar Card
A8- Driving License of the Complainant,  
A9- Estimate of Damages

PRAYER:
The complainant therefore prays to direct the opposite party:-
To Pay Rs.5,85,762/- towards estimated damages;
To Pay Rs.3,60,000/- towards loss of earnings i.e. Rs. 30,000/- p.m x 12 months from March, 2020 to February 2021; 
To Pay interest at the rate of 24% p.a. on Rs. 9,45,762 (on the above total amount) from the date of the accident i.e, from 22/02/2020 to till date of realization of the amount;
To Pay Rs.10,000/- towards damages of the goods in the vehicle;
To Pay Rs. 2,00,000/- towards Compensation;
To Pay Rs. 1,00,000/- towards costs of the complainant;
Such other relief or reliefs which the Honourable Forum deemed fit, just and proper in the circumstances of the case.


PLACE:                                                                                                                 Signature
DATE:



VERIFICATION:
I , Nammi Venkata Rao s/o Ramulu (late) aged 47 years is a resident of  D. No 9-40-22, Nammivanipeta, Chittivalasa Post, Bheemunipatnam Mandal, Visakhapatnam District- 531162 hereby declare that I have not misrepresented any facts nor have tried to hide any information in my above complaint. All the facts mentioned herein are true to the best of my knowledge.

Name & signature of the complainant



User: Yes, Please. It would be really helpful 

Consumer Grievance Assistance Chatbot:  Sure. Here is a draft Affidavit. and Memorandum of parties.. Please print and verify the names and addresses on the memorandum of parties. For the affidavit, please sign it before a notary public and have the affidavit notarized. Both documents should be scanned and uploaded to the e-daakhil portal. Do you have any other questions or is there more information you'd like?
BEFORE THE DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION NO.0I VISAKHAPATNAM: AP

CONSUMER COMPLAINT NO.       /2021

IN THE MATTER OF: 
Nammi Venkata Rao
S/o Ramulu (Late), aged 47 years,
D. No 9-40-22, Nammivanipeta, 
Chittivalasa Post, Bheemunipatnam Mandal,
Visakhapatnam District- 531162
Occupation: Agriculture
… COMPLAINANT
AND
The Cholamandalam MS General 
Insurance Company Limited, 
represented by its Authorized Person, 
registered and Head Office, 2nd Floor, 
Dare House, 2 NSC Bose Road, 
Chennai-600001, Tamil Nadu
The Cholamandalam MS General 
Insurance Company Limited, 
represented by its Authorized Person,
Branch Office, H.No 40-1-48/1, 2nd Floor, 
Beside Centro, Opp Airtel Office, 
Near DV Manor Hotel, MG Road, 
Vijayawada-520010, Krishna (Dist), 
Andhra Pradesh
Cholamandalam MS General 
Insurance Company Limited, 
represented by its Authorized Person, 
D. No 26-1-19, Opp MSR High Tech 
Bus Stand Kambala Peta, 
Rajahmundry- 533103, East Godavari Dist,
Andhra Pradesh
… OPPOSITE PARTIES




BEFORE THE DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION NO.0I VISAKHAPATNAM: AP

CONSUMER COMPLAINT NO.       /2021

IN THE MATTER OF: 
Nammi Venkata Rao
S/o Ramulu (Late), aged 47 years,
D. No 9-40-22, Nammivanipeta, 
Chittivalasa Post, Bheemunipatnam Mandal,
Visakhapatnam District- 531162
Occupation: Agriculture
… COMPLAINANT
AND
The Cholamandalam MS General 
Insurance Company Limited, 
represented by its Authorized Person, 
registered and Head Office, 2nd Floor, 
Dare House, 2 NSC Bose Road, 
Chennai-600001, Tamil Nadu
The Cholamandalam MS General 
Insurance Company Limited, 
represented by its Authorized Person,
Branch Office, H.No 40-1-48/1, 2nd Floor, 
Beside Centro, Opp Airtel Office, 
Near DV Manor Hotel, MG Road, 
Vijayawada-520010, Krishna (Dist), 
Andhra Pradesh
Cholamandalam MS General 
Insurance Company Limited, 
represented by its Authorized Person, 
D. No 26-1-19, Opp MSR High Tech 
Bus Stand Kambala Peta, 
Rajahmundry- 533103, East Godavari Dist,
Andhra Pradesh
… OPPOSITE PARTIES




AFFIDAVIT

I, Nammi Venkata Rao S/o Ramulu (Late), aged 47 years, and residing at D. No 9-40-22, Nammivanipeta, Chittivalasa Post, Bheemunipatnam Mandal, Visakhapatnam District- 531162, do hereby solemnly affirm and declare as under:

That the facts stated in the complaint from paras 1 to 15, in Nammi Venkata Rao Vs. The Cholamandalam MS General Insurance Company Limited and Ors, being filed before the Hon’ble District Consumer Disputes Redressal Commission No.0I, Visakhapatnam: AP are true to the best of my knowledge and based on the records maintained by me, which I believe to be true.

Deponent



VERIFICATION

Verified at, on this ___________ day that the contents of the above affidavit are true and correct to the best of my knowledge and belief.


Deponent



User: No, that's all for now. Thanks.

Consumer Grievance Assistance Chatbot: Thank you for using the tool. Feel free to raise further queries anytime or you can log on to  https://consumeraffairs.nic.in/ or dial 1800-11-4000 (National Consumer Helpline) for more information.
'''

one_shot_prompt = "Here is an example of how the chatbot would assist with consumer grievance. Adapt language, tone, and level of detail to match variations in user submissions while maintaining the core structure. Use it for only styling purposes. This is a demo conversation. \n Example: \n "+one_shot_example


GENERAL_CORPUS = '''
Question: “When was the Consumer Protection Act 2019 enacted? Is the Consumer Protection Act 1986 still valid?” 

Answer: “The Consumer Protection Act 2019 was notified on August 9, 2019. However, it came into effect on July 20, 2020. This Act replaced the Consumer Protection Act 1986 to address the contemporary issues faced by consumers and to streamline the consumer grievance redressal process in India. 

The Consumer Protection Act 1986 was the first significant legislation in India aimed at protecting consumer interests. It provided for the establishment of consumer councils and other authorities for the settlement of consumers' disputes and for matters connected therewith. It has now been repealed by the Consumer Protection Act 2019. Under Section 107(2) of the Consumer Protection Act 2019, all disputes arising out of the previous act shall now be governed by the new act.”

—

Question: “What are the main features of the Consumer Protection Act 2019?”
Answer: “The Consumer Protection Act 2019 retains several provisions from the older legislation but introduces numerous new measures to strengthen consumer rights and provide a comprehensive framework for consumer protection. Here are the main features of the Act:
Inclusion of E-commerce and Direct Selling Transactions: The Act now covers transactions conducted through e-commerce platforms and direct selling. This ensures that consumers engaging in online and direct sales are protected under the law.
Establishment of Central Consumer Protection Authority (CCPA): A key feature of the new Act is the establishment of the Central Consumer Protection Authority (CCPA). The CCPA has been empowered to regulate matters relating to the violation of consumer rights, unfair trade practices, and misleading advertisements. It can take suo-moto actions, recall products, and initiate class actions.
Strict Norms for Misleading Advertisements: The Act imposes strict regulations on misleading advertisements. It holds manufacturers, service providers, and endorsers accountable for false claims and deceptive marketing practices, thus protecting consumers from being misled.
Product Liability: The new Act introduces stringent norms for product liability, making manufacturers, sellers, and service providers responsible for any harm caused by defective products or deficient services. This ensures that consumers can seek compensation for any damage or injury.
Enhancement in Pecuniary Jurisdiction: The Act has increased the pecuniary jurisdiction of consumer courts. This means that the monetary limit for cases that can be handled by District Consumer Disputes Redressal Commission (DCDRC), State, and National Consumer Disputes Redressal Commission (NCDRC) has been raised, allowing more significant claims to be addressed at each level.
Greater Ease in Dispute Resolution: The Act aims to simplify and expedite the dispute resolution process. It promotes the resolution of consumer disputes through mediation, offering an alternative to the traditional court process. This makes it easier and faster for consumers to settle their grievances.
Addition of Unfair Trade Practice Clauses: The Act has expanded the definition of unfair trade practices to include unfair contracts. This ensures that any terms in a contract that are biased against the consumer or are excessively one-sided can be challenged and declared null and void.
E-filing of Complaints: Consumers can now file complaints electronically, making the process more accessible and convenient. This is particularly beneficial for those who may not be able to physically visit consumer courts.
Mediation as an Alternate Dispute Resolution: The introduction of mediation as an alternate dispute resolution mechanism provides a more amicable and less adversarial way of resolving consumer disputes. This helps in reducing the burden on consumer courts and provides quicker resolutions.”
—
Questions: “What Rights are guaranteed under Consumer Protection Act, 2019 for consumers?” 

“The Consumer Protection Act of 2019 ensures several fundamental rights for consumers. These rights are crucial pillars in safeguarding consumer interests and ensuring fair treatment in the marketplace. Here are the six rights guaranteed under the Act:

1. Right to Safety: Consumers have the right to expect that the products and services they purchase are safe for use or consumption. This right ensures protection from hazardous or substandard goods.

2. **Right to be Informed**: Consumers have the right to receive accurate and transparent information about the products and services they intend to purchase. This includes details regarding quality, quantity, price, ingredients, and any associated risks.

3. **Right to Choose**: Consumers have the freedom to select from a range of products and services offered by various providers. This right promotes competition in the market and empowers consumers to make informed decisions based on their preferences.

4. **Right to be Heard**: Consumers have the right to express their grievances and concerns regarding unsatisfactory products or services. This includes the opportunity to voice complaints, provide feedback, and seek resolution through accessible channels.

5. **Right to Seek Redressal**: Consumers have the right to seek appropriate remedies and compensation for any harm or inconvenience caused by defective products or deficient services. This right ensures access to fair and efficient dispute resolution mechanisms.

6. **Right to Consumer Awareness**: Consumers have the right to access information and resources that enhance their understanding of their rights and responsibilities in the marketplace. This includes education on consumer protection laws, awareness campaigns, and access to relevant support services.

These rights collectively empower consumers to make informed choices, demand accountability from businesses, and seek recourse in cases of unfair treatment or exploitation. By upholding these rights, the Consumer Protection Act aims to foster trust and confidence in the marketplace while promoting a fair and competitive economic environment.” 

—
Question: “Who is a Consumer?”
Answer: “Under the Consumer Protection Act 2019, a consumer is defined comprehensively to include individuals engaging in various types of transactions. Here's an expanded explanation:
Definition of a Consumer:
General Definition:
A person who buys any goods: This includes anyone who purchases goods for consideration (payment), whether the payment is made upfront, promised, partly paid, or deferred. The definition also extends to users who use the goods with the buyer's approval and beneficiaries of services.
Hires or avails any services: Similar to the purchase of goods, this includes individuals who hire or use services for consideration, whether the payment is made, promised, partly paid, or deferred.
Inclusion of Various Transaction Modes:
The Consumer Protection Act 2019 explicitly includes both offline and online transactions. This encompasses purchases made through electronic means such as websites and mobile apps, as well as teleshopping, direct selling, and multi-level marketing.
Personal Use Criterion:
To qualify as a consumer under the Act, the goods or services must be bought for personal use and not for commercial purposes like manufacturing or resale. This distinction ensures that the Act primarily protects individuals who are end-users of products and services, rather than businesses engaged in commercial activities.
Examples of a Consumer:
Individual Purchaser: Someone who buys a refrigerator for use in their home.
Service User: A person who subscribes to an internet service for personal use.
Online Shopper: An individual who purchases clothing from an e-commerce website.
Teleshopping Buyer: Someone who orders a fitness equipment via a teleshopping channel.
Direct Selling: A person who buys skincare products directly from a company representative.
Multi-Level Marketing: An individual purchasing wellness products through a multi-level marketing scheme for personal use.”
—
Question: “Who is not a consumer?”
Answer: “A consumer is defined as someone who buys goods or avails services for personal use and not for commercial purposes. However, certain categories of people are explicitly excluded from this definition. Here are the main exclusions:
A person who obtains goods free of charge: If an individual receives goods without any payment, they are not considered a consumer under the Act. This means they cannot seek remedies under consumer protection laws for issues related to those goods.
A person who avails services free of charge: If someone uses services without paying for them, they do not fall under the definition of a consumer. Consequently, they cannot file complaints or seek redressal for any grievances related to those services. However, if services are provided for free along with a product, you can argue that since you paid for the product, those services should be included in a product liability claim.
A person who avails services for any commercial purposes: Those who use services for commercial purposes, such as in their business operations, are also excluded from the definition of a consumer. The protection offered by the Act is intended for personal, not business, use.
A person who avails services under a contract of service: Individuals who receive services as part of a contract of service, such as employees receiving services from their employers, are not considered consumers. These situations are typically governed by employment laws rather than consumer protection laws.
Clarification on commercial purposes: It is important to note that the Act specifies an exception to the exclusion for commercial purposes. If a person buys and uses goods exclusively for the purpose of earning their livelihood through self-employment, they are still considered a consumer. This means that small business owners, artisans, or individuals using goods for personal business activities to sustain their livelihood are protected under the Act.”
—
Question: “Are individuals who purchase goods and resell them classified as consumers?”
Answer: “ Individuals or entities who purchase goods to resell them or use them for commercial purposes are not classified as consumers. Consumer specifically refers to end-users who purchase goods or services for personal consumption or use, not for commercial gain through resale or business operations.”

—
Question: “What are goods?”

Answer: “Goods refer to every kind of movable property. This definition encompasses all items that can be physically moved, regardless of their nature or purpose. Additionally, "goods" explicitly include "food" as defined in clause (j) of sub-section (1) of section 3 of the Food Safety and Standards Act, 2006. This means that all consumable items classified as food under this Act are also considered goods. However, immovable items such as property are not covered under goods”
_

Question: “What is deficiency under the Consumer Protection Act, 2019?”
Answer: “Deficiency" under the Consumer Protection Act refers to any fault, imperfection, shortcoming, or inadequacy in the quality, nature, and manner of performance that a person is required to maintain by law or has undertaken to perform in relation to any service. This includes:
Negligence or Omission: Any act of negligence, omission, or commission by the service provider that causes loss or injury to the consumer.
Withholding Information: Deliberate withholding of relevant information by the service provider to the consumer.
Explanation: Whenever there is a deficiency in services, the customer is taken advantage of, leading to financial loss. Any form of negligence, omission, or wrongful act can result in harm to consumers.
Section 2(11) of the Consumer Protection Act, 2019: This section defines "deficiency in service" in a similar manner. It states that any fault, imperfection, shortcoming, or inadequacy in the quality, nature, and manner of performance that is required to be maintained by law or as per a contract is considered a deficiency. This includes any act of negligence, omission, commission, or deliberate withholding of relevant information by the service provider to the consumer.
Applicable Sectors: Deficiency of service can occur in any buyer-seller relationship and spans across various sectors including legal aid, banks, railways, construction, transportation, education, electricity, entertainment, restaurants, and hospitality.
In summary, a deficiency in service involves any failure to meet the required standards or contractual obligations, causing harm or loss to the consumer.”
—
Question: “What is an unfair contract?”
Answer: “An unfair contract refers to a contract between a manufacturer, trader, or service provider and a consumer that contains terms causing a significant change in the consumer's rights. These terms include:
a. Excessive Security Deposits: Requiring consumers to provide manifestly excessive security deposits for the performance of contractual obligations.
b. Disproportionate Penalties: Imposing penalties on consumers for breach of contract that are wholly disproportionate to the loss incurred by the other party.
c. Refusal of Early Repayment: Refusing to accept early repayment of debts even when the consumer is willing to pay the applicable penalty.
d. Unilateral Termination: Entitling one party to terminate the contract unilaterally without a reasonable cause.
e. Detrimental Assignment: Permitting one party to assign the contract to the detriment of the consumer without the consumer's consent.
f. Unreasonable Charges or Conditions: Imposing any unreasonable charge, obligation, or condition on the consumer that puts them at a disadvantage.
In summary, an unfair contract contains terms that significantly disadvantage the consumer, either through excessive demands, disproportionate penalties, or unbalanced rights and obligations.”
_
Question: “What are the unfair trade practices under the Consumer Protection Act, 2019?”
Answer: “The Consumer Protection Act, 2019, prohibits various unfair trade practices, including:
False or Misleading Advertisements: Advertising goods or services with false or misleading claims.
Selling Defective or Unfit Goods: Selling goods that are defective or unfit for their intended purpose.
Excessive Pricing: Charging excessively high prices for goods or services.
Refusal to Refund or Replace: Refusing to refund money or replace defective goods as per the terms of sale or service.
Coercion: Using coercion, undue influence, or pressure to compel consumers to purchase goods or services.
Unfair Contract Terms: Including unfair terms and conditions in contracts with consumers.
These practices are deemed unfair as they exploit consumers and violate their rights, as outlined in the Consumer Protection Act, 2019.”
—
Question: “What is a misleading advertisement ?”
Answer: “A misleading advertisement, concerning any product or service, is one that:
Falsely describes the product or service,
Provides false guarantees or is likely to mislead consumers regarding its nature, substance, quantity, or quality,
Conveys explicit or implicit representations that, if made by the manufacturer, seller, or service provider, would be considered unfair trade practices, or
Intentionally hides significant information.” 
—
Question: “What is e-commerce?”
Answer: “E-commerce, as defined under the Consumer Protection Act 2019, refers to the buying or selling of goods or services, including digital products, over a digital or electronic network. This includes transactions conducted through online platforms, websites, mobile apps, or other electronic means where goods or services are exchanged electronically.”
_
Question: “What is an e-commerce entity” 
Answer: “An "e-commerce entity" refers to any individual who owns, operates, or manages a digital or electronic platform for electronic commerce. However, this term does not encompass a seller who offers goods or services for sale on a marketplace operated by an e-commerce entity.” 
—
Question: “What is product liability?”
Answer: “Product liability refers to the responsibility of a product manufacturer or seller to compensate a consumer for any harm caused by a defective product or deficiency in services related to the product. This means that if a product is faulty or does not meet the expected standards, and this results in harm to a consumer, the manufacturer or seller is liable to provide compensation. Product liability ensures that consumers are protected from unsafe or substandard products and have recourse if they suffer harm as a result of using them.”
—
Question: “ What is product liability action?”
Answer: “A "product liability action" refers to a complaint lodged by an individual before a District Consumer Disputes Redressal Commission (DCDRC), State Commission, or National Consumer Disputes Redressal Commission (NCDRC), seeking compensation for harm suffered. To initiate such action, the consumer must demonstrate that the harm resulted from a "defective product or service" purchased by them.” 
—
Question: “What is harm under the Consumer Protection Act, 2019?”
"Harm in the context of product liability includes:
Damage to Property: This refers to any damage caused to property other than the product itself. For example, if a faulty electronic device causes a fire that damages a house, this would be considered harm.
Personal Injury, Illness, or Death: Harm also includes any personal injury, illness, or death caused by a defective product. For instance, if a toy contains hazardous materials that harm a child, this would be considered harm.
Mental Agony or Emotional Distress: This refers to the mental anguish or emotional distress experienced as a result of personal injury, illness, or damage to property caused by a defective product.
Loss of Consortium or Services: Harm also includes any loss of consortium (companionship or support) or services resulting from the harm caused by a defective product.
Exclusions: Harm does not include harm caused to the product itself or damage to property due to breach of warranty conditions. It also does not include any commercial or economic loss, including direct, incidental, or consequential loss relating to the harm.”
_
Question: “Who is a product seller ?” 
Answer: “A product seller is a person engaged in the business of importing, selling, distributing, leasing, installing, preparing, packaging, labelling, marketing, repairing, maintaining, or otherwise commercially placing a product. This includes:
- Manufacturers who also sell products,
- Service providers, excluding:
 Sellers of immovable property, except those engaged in selling constructed houses or constructing homes or flats,
Providers of professional services where the sale or use of a product is incidental, with the essence of the transaction being the provision of opinion, skill, or services,
Persons who:
Solely act in a financial capacity regarding the product sale,
Are not manufacturers, wholesalers, distributors, retailers, direct sellers, or electronic service providers,
Lease a product without having a reasonable opportunity to inspect for defects, under a lease arrangement where selection, possession, maintenance, and operation are controlled by someone other than the lessor.” 
—
Question: “What are the Consumer Dispute Redressal Agencies?”

Answer: “ Consumer Disputes Redressal Agencies, established under the Consumer Protection Act, are quasi-judicial bodies offering prompt and affordable resolution to consumer grievances. They operate at three levels: District Consumer Disputes Redressal Commission (DCDRC), State, and National Consumer Disputes Redressal Commission (NCDRC), known respectively as District Consumer Disputes Redressal Commission (DCDRC), State Consumer Disputes Redressal Commission (SCDRC), and National Consumer Disputes Redressal Commission (NCDRC).” 
_
Question: “What is CCPA?”
Answer: “CCPA stands for the Central Consumer Protection Authority. It is an institution established by the Central Government under the provisions of the Consumer Protection Act to regulate various aspects related to consumer rights and protection. The CCPA has been granted powers to:
Regulate Matters: The CCPA regulates matters pertaining to the violation of consumer rights, unfair trade practices, and false or misleading advertisements that are detrimental to the interests of the public and consumers.
Protect Consumer Rights: The primary objective of the CCPA is to promote, protect, and enforce the rights of consumers as a collective class. This includes safeguarding consumers against deceptive practices, ensuring fair treatment, and upholding their rights in commercial transactions.A complaint about violations of consumer rights, unfair trade practices, or misleading advertisements can be made to the District Collector, Commissioner of a regional office, or the Central Authority. Grievances Against Misleading Advertisements (GAMA) - You can register a complaint along with a copy / video / audio of such advertisement through the web portal of the GOI at http://gama.gov.in to bring it to the notice of the Government” 
Consumers with individual grievances cannot approach the CCPA for addressing consumer grievances as it is only a regulatory authority. 
—
Question: “Can a person buying goods or services for a business purpose make a complaint ?”

Answer: “ Under the Consumer Protection Act of 2019, individuals purchasing goods or services for business purposes cannot file complaints unless they are buying goods or using services solely for the purpose of self-employment for their livelihood. 
Any goods purchased in the course of business on behalf of their employer or company in which they have an interest, would not be covered under consumer protection. Similarly, any goods or services purchased in pursuit of a principal client relationship on behalf of a business would also not be covered by consumer protection. For example, if a person buys 10 laptops from a manufacturer on behalf of his company or place of work, that would be buying goods for a business purpose. Any issue arising out of this purchase will not be covered under the Consumer Protection Act, 2019.
However, if the goods or services are purchased in the course of self-employment for pursuit of livelihood, consumer protection law will apply to the goods and/or services purchased. For example, a self-employed carpenter purchasing a drill for his work would be able to make use of consumer protection law with respect to his purchase of the drill.”
—
Question: “Can someone availing of public services make a complaint against the government under Consumer Protection Act?”
Answer: When the government provides services as a public authority, such as delivering public services to all citizens, no claims can be made under consumer protection law. This is because the government is not considered a seller or service provider in these cases. For example, a person cannot use consumer protection laws to address issues like delayed Income Tax refunds or dissatisfaction with the work of a municipal authority.
However, if services or goods are provided by a government-owned entity and a fee is paid for these services or goods, then consumer protection laws can apply. In such cases, the government-owned entity acts as a seller or service provider. For example, if someone has paid for services like railway tickets or purchased goods from a government-run shop, they can file a complaint under consumer protection law for any deficiencies or issues with the service or goods provided.
—
Question: “Can I claim compensation if the product itself is damaged?”

Answer: “ No, compensation cannot be claimed solely for damage to the product itself under product liability.” 
—
Question: “Who can be liable in a product liability action?”
You can file a product liability claim against various parties involved in the supply chain, including:
Manufacturer: The entity responsible for producing the product.
Product Service Provider: Any entity that provides a service related to the product, such as installation or maintenance.
Product Seller: The entity that sells the product to consumers.
Depending on the circumstances of the case, you may choose to hold one or more of these parties responsible for any harm caused by the product.”
—
Question: “What grounds can I claim compensation for the product manufacturer?

Answer: “You can claim compensation from a product manufacturer if, in a product liability action under Section 84 of the Consumer Protection Act, 2019:

1. The product has:
   a. A manufacturing defect,
   b. A design defect,
   c. Deviation from manufacturing specifications,
   d. Failed to conform to the express warranty, or
   e. Lacked adequate instructions or warnings regarding proper usage to prevent harm or misuse.

2. The manufacturer can still be held liable in a product liability action even if they prove they were not negligent or fraudulent in providing an express warranty for the product.”
—
Question: “ What grounds can I claim compensation for a product service provider?”
Answer: “A product service provider can be held liable in a product liability action if:
a. Faulty, Imperfect, or Deficient Service: The service provided was faulty, imperfect, deficient, or inadequate in quality, nature, or manner of performance as required by law, contract, or otherwise.
b. Act of Omission, Commission, or Negligence: The service provider committed an act of omission, commission, negligence, or consciously withheld information that caused harm.
c. Failure to Provide Adequate Instructions or Warnings: The service provider did not issue adequate instructions or warnings to prevent harm.
d. Non-Conformity to Express Warranty or Contract Terms: The service did not conform to express warranty or the terms and conditions of the contract.
In these situations, the product service provider can be held liable for any harm caused to the consumer.”
—
Question: “ When can I claim compensation from the seller?”
Answer: “ You can claim compensation from a seller who is not the manufacturer in a product liability action if:
The seller exercised substantial control over the designing, testing, manufacturing, packaging, or labeling of the product that caused harm.
The seller altered or modified the product, and such alteration or modification was a substantial factor in causing harm.
 The seller provided an express warranty for the product independent of any warranty made by the manufacturer, and the product failed to meet the seller's warranty, resulting in harm.
The product was sold by the seller, but the identity of the manufacturer is unknown, or if known, legal processes cannot be served against them.
The seller is not subject to Indian law, or any orders passed cannot be enforced against them.
The seller failed to exercise reasonable care in assembling, inspecting, or maintaining the product, or failed to provide warnings or instructions from the manufacturer regarding product dangers or proper usage, and this failure directly caused harm.” 
—

Question: “What are the defences available to a defendant in a product liability action?”
Answer: “In a product liability action, the following defences may be available to the defendant:
Non-Consumer Status: The complainant is not considered a 'consumer' under the Consumer Protection Act 2019, meaning they obtained the goods for resale or commercial purposes, or received the product free of charge.
Misuse, Alteration, or Modification: The product was misused, altered, or modified by the consumer, which contributed to the harm.
Obvious or Commonly Known Danger: The danger posed by the product is obvious or commonly known to the user or consumer, or should have been known considering the characteristics of the product.
Warnings or Instructions: The product manufacturer provided warnings or instructions to the employer of the consumer, and the harm resulted from the employer's use of the product.
Component of End Product: The product was sold as a component of an end product, and necessary warnings or instructions were provided by the manufacturer, but the harm was caused by the end product's use.
Expert Supervision: The product was legally meant to be used or dispensed only by or under the supervision of an expert, and the manufacturer provided reasonable warnings or instructions to such expert or class of experts.
Consumer's State While Using the Product: The consumer was under the influence of alcohol or a prescription drug not prescribed by a medical practitioner while using the product.
Lack of Harm: Even if the product was defective, it did not cause any harm to the consumer who used it.”
Question: “ Under what conditions is a “marketplace” responsible for product liability?” 
Answer: “ A "marketplace" is generally not responsible for product liability issues if it merely offers a platform for buying and selling products. However, according to the CPA 2019, a marketplace may be liable in a product liability claim if it provides extra services or functions as a product manufacturer or service provider. Examples include situations where the marketplace:
a) Offers its own separate warranty for a product,
b) Provides additional services, such as installation, for a product.”
—
Question: “Who is responsible if several parties are involved in the manufacturing of a product?”
Answer: “ In cases involving multiple manufacturers, liability is determined by the commission on a case-by-case basis. Factors considered include which component caused the defect and whether proper instructions or warnings were provided. This is especially relevant for products like home appliances and automobiles with parts from different manufacturers. Courts have attempted to identify the responsible party in past cases with multiple parties involved. If assigning liability to a single party is not possible, all involved parties may be held jointly and severally liable.”
—
Question: “Who can be held liable for a misleading advertisement?”
Answer: “Any of the following parties can be held liable for a misleading advertisement:
Manufacturers: The producers of the product being advertised.
Traders: Individuals or businesses involved in the sale and distribution of the product.
Advertising Agencies: Agencies responsible for creating and promoting the advertisement.
Celebrity Endorsers: Celebrities or public figures who endorse and promote the product.
Publishers: Media outlets and platforms that disseminate the advertisement to the public.
These parties can be held accountable for any false claims, deceptive practices, or misleading information presented in advertisements.”
—
Question: “What is the penalty for someone found guilty of the offence of misleading advertisement?” 
Answer: “For the offence of misleading advertisement, if found detrimental to consumer interests, the Central Authority can levy a penalty of up to Rs. 10,00,000 (Rupees Ten Lakhs) on the manufacturer, escalating to Rs. 50,00,000 (Rupees Fifty Lakhs) for subsequent violations. Moreover, under the Consumer Protection Act 2019, manufacturing or service providers engaging in false advertising can face imprisonment for up to 2 years, extending to 5 years for repeat offences.
Similarly, the Central Authority can impose a penalty of up to Rs. 10,00,000 (Rupees Ten Lakhs) on the endorser if the misleading advertisement harms consumer interests. Additionally, the endorser may be barred from endorsing the specific product or service for up to one year, which could extend to three years for subsequent violations. However, endorsers exercising due diligence to verify the advertisement's claims regarding the endorsed product or service are not liable to penalties.
— 
——

Question: “Before resorting to litigation, is there any process that can be pursued in order to claim compensation?”
Answer: “Yes, before resorting to litigation, an aggrieved individual can take the following steps to claim compensation:
Send a Notice: The individual can send a formal notice to the service provider or seller. This notice should detail:
The defects in the goods or services or the deficiencies in the service
The individual's intention to pursue legal action if the service provider fails to offer compensation
This step allows the service provider an opportunity to rectify the issue or offer compensation, potentially resolving the dispute without the need for litigation.”
_
Questions: “Can I be asked to waive the claim under the Act?” 
Answer: “ Waiving a claim under the Consumer Protection Act, 2019 (CPA 2019), may be deemed unfair as per the broad definition of unfair contracts outlined in the Act. Additionally, agreements that restrict legal proceedings, such as those preventing parties from enforcing their rights or limiting the time to do so, are void under the Indian Contract Act, 1872, and thus unenforceable. While the issue of waiving claims has not been tested in court, contractual language regarding such waivers should be carefully examined to avoid potential unenforceability under both the CPA 2019 and Indian contract law.” 

—
Question: “How do I get my consumer grievances addressed?”
Answer:  “To address your consumer grievance, you have a few options. You can:

Directly send a notice to the product seller or service provider.
Call the National Consumer Helpline at 1800-11-4000 or 1915, available all days except national holidays, from 8:00 AM to 8:00 PM.
Send an SMS to 8800001915, and they will get back to you.
Other helpline numbers are - 011 - 2430 - 5702, 011 - 2430 - 5703, 011 - 2430 - 5728, 011 - 2430 - 5752, 011 - 2430 - 5766
Register your grievance online at https://consumerhelpline.gov.in/user/signup.php.
Register your grievance through the NCH APP or UMANG APP on your mobile phone.
Register your grievance via WhatsApp at https://api.whatsapp.com/send/?phone=918800001915&text&type=phone_number&app_absent=0.
The new platform is e-Jagriti, where users can file complaints online.The link is https://e-jagriti.gov.in/ 

—
Question: “How can I contact a state consumer helpline?”
Answer: “The state helpline numbers for any consumer grievance related assistance is below:
Andhra Pradesh - 0866 2551431

Arunachal Pradesh - 1800-345-3601

Assam - 1800-345-3611

Bihar - 1800-345-6188

Chhattisgarh - 1800-233-3663

Delhi - 011-23379266

Goa - 0832-2222466

Gujarat - 1800-233-0222

Haryana - 1800-180-2087

Himachal Pradesh - 0177-2620797

Karnataka -  080-22355065

Kerala - 0471-2725157

Madhya Pradesh - 0755-2553722, 2763673

Maharashtra - 1800-22-2262

Meghalaya - 1967 or 1800-345-36870

Mizoram -  1800-354-3891

Odisha - 1800-345-6724 / 6760

Rajasthan - 1800-180-6030

Sikkim - 1800-345-3209/203-951

Tamil Nadu - 044-2859-2828

Telangana - 040-23394399

Uttarakhand - 0135-3510041
West Bengal - 1800-345-2802”
—
Question: “What is the address or contact information of state consumer commissions in India?”
Answer: “The different state consumer commissions address is as below:
Andhra Pradesh -1 st Floor, Sai Towers, Beside SIRIS Company, Ashok Nagar, Kanuru, Vijayawada – 520 007
Arunachal Pradesh -State Consumer Disputes Redressal Commission, Near Pawan Hans Office, (Old Secretariat Complex), Naharlagun, ITANAGAR
Assam -Housefed Complex, Central Block, 5th Floor,Guwahati – 781 006
Bihar- State Consumer Disputes Redressal Commission, R – Block, Road No.2, Daroga Prasad Rai Path,PATNA – 800 001
Chhattisgarh- State Consumer Disputes Redressal Commission,Behind Bus Stand (Old Bus Depot No.1 Complex)Pandri,RAIPUR - 492 004
Delhi- State Consumer Disputes Redressal Commission A-Block, First Floor, Vikas Bhavan, I.P. Estate, New Delhi-110002
Goa- Goa State Consumer Disputes Redressal Commission, 1st lift, 4th floor, Junta House, Panaji Goa- 403001
Gujarat- Consumer Disputes Redressal Commission, 'Grahak Bhavan' Nr. Gota Cross Road, S.G. Highway, Ahmedabad -380 060
Haryana- Bays No. 3 - 6, Sector – 4, Panchkula – 134 109 (Haryana)
Himachal Pradesh- S. D. A. Complex, Block No. 33, Shimla -171009
Jammu & Kashmir- 3, Manda Hill, Rehari, Jammu Tawi, Jammu & Kashmir
Jharkhand- State Consumer Disputes Redressal Commission, Near High Court, Doranda Bazar, Doranda, RANCHI – 834 002.
Karnataka- Basava Bhavana, High Grounds,Bengaluru-560001
Kerala - State Consumer Disputes Redressal Commission Kerala, isuvihar Lane, Vazhuthacaud, Thiruvananthapuram-695010.
Madhya Pradesh- M.P.State Consumer Disputes Redressal Commission Plot no.76, Arera Hills, Bhopal PIN-462011
Maharashtra- Head Office - Old Secretariat Building, Ground Floor,   Near Jahagir Art Gallery M. G. Road, Behind Elphiston College, Mumbai – 400 032
Manipur- Old Assembly Complex, Imphal, West District, Manipur - 795 001
Meghalaya- Commission Meghalaya, Horse Shoe Building, Lower Luchumiere, Shillong. District: East Khasi Hills - 793001
Mizoram-  room no. 228, second floor, building no. 2. Mizoram secretariat, New capital complex (MINECO) khatla, Aizawl, Mizoram
Nagaland- Nagaland State Consumer Disputes Redressal Commission,Directorate of Legal Metrology & Consumer Protection
Odisha- Cuttack, C.D.A sector, 1, near sati Choura chowk, bidanasi, Cuttack-753014
Punjab- Plot No. 1037, Sector – 37A,Chandigarh – 160 036
Rajasthan- handloom haveli, Ashok marg, c- scheme , jaipur -302001
Sikkim- ATTI Building,Kazi Road,Gangtok,East Sikkim
Tamil Nadu- Chennai , TNPSC road, park town Chennai -600 003
Telangana- Eruvaka Building, Khairatabad, Hyderabad – 500 004.
Tripura- P. N. Complex, Gurkhabasti, Agartala West Tripura, - 799006
Uttar Pradesh- C – 1, Vikrant Block – 1, Near Shaheed Path, Gomati Nagar, - 226 010
Uttarakhand- 23/16, Circular Road,Dalanwala, Dehradun- 248 001
West Bengal- Ground Floor,11A, Mirza Ghalib Street,Kolkata - 700087”

—
Question: “What are the other remedies available for consumers for harm caused to them?”
Answer: “In addition to filing a complaint before the Consumer Commission, consumers have the option to seek remedies through sectoral regulators. Depending on the sector, consumers can approach the following bodies:
Airlines: Director General of Civil Aviation’s Air Sewa Portal (https://airsewa.gov.in/grievance/grievance-redressal)
Banking: RBI Banking Ombudsman (https://rbi.org.in/Scripts/Complaints.aspx)
Broadband & Internet: Telecom Consumer Complaints Monitoring System (TCCMS) (https://tccms.trai.gov.in/Queries.aspx?cid=1)
Bureau of Indian Standards (BIS): For complaints related to standards of products.
Drugs and Cosmetics: Drug Controller General of India (https://cdsco.gov.in/opencms/opencms/en/About-us/who/)
DTH & Cable Services: Email to ibf@ibfindia.com for general issues and ibf@ibfindia.com for content issues.
Electricity Services: Each state has its regulatory authority. For a list of these, visit https://jercjkl.nic.in/otherjercandserc.html
Financial Services: RBI Banking Ombudsman (https://rbi.org.in/Scripts/Complaints.aspx)
Food and PDS Schemes: https://www.fssai.gov.in/cms/grievance.php
Grievances Against Misleading Advertisements (GAMA): Register a complaint along with a copy/video/audio of the advertisement through the web portal of the Government of India at http://gama.gov.in
Insurance: Use the Bima Bharosa system on the IRDAI Portal (https://bimabharosa.irdai.gov.in/) for registering and monitoring complaints. Email complaints can be sent to complaints@irdai.gov.in. Toll-Free No.: 155255 or 1800 4254 732.
Legal Metrology: Lodge a complaint with the District Legal Metrology Officer or the Commissioner of the respective states. For more information, visit http://consumeraffairs.nic.in/forms/contentpage.aspx?lid=639
Non-Banking Financial Companies (NBFCs): RBI Banking Ombudsman (https://rbi.org.in/Scripts/Complaints.aspx)
Pension Funds: Central Pension Accounting Office (https://cpao.nic.in/grievance_sql/Grievance_form_all.php)
Telecom: Telecom Consumer Complaints Monitoring System (TCCMS) (https://tccms.trai.gov.in/Queries.aspx?cid=1)
Consumers can explore these avenues to seek redressal for grievances related to specific sectors.”
—
Question: “Can I file a consumer complaint using my phone?”
Answer: “Yes, you can file a consumer complaint online using your phone. One way to do this is by using the UMANG App, which is the National Consumer Helpline’s (NCH) application. Here's how you can file a complaint using the UMANG App:
Download UMANG App: Visit the Google Play Store or the Apple App Store and download the UMANG App.
Log In or Sign Up: Log in if you already have an account, or sign up for a new one.
Search for National Consumer Helpline: In the All Services section of the app, search for the National Consumer Helpline.
Register Grievance: Press the Register Grievance button.
Provide Details: Provide information such as the state and city where the product was purchased, the industry, the firm name, and a description of the grievance.
Submit Complaint: Submit your complaint through the app.
Please note that the process may vary slightly based on the specific features and interface of the UMANG App.”
—
Question: “What is the pecuniary Jurisdiction of the Consumer commissions in India ?”

Answer: “The pecuniary jurisdiction of each level of the Consumer Commissions in India varies according to the total value of good or services along with the value of the claimed compensation:
District Consumer Disputes Redressal Commission (DCDRC): Handles cases where the total value claimed (i.e. total value of goods or services + value of compensation claimed) does not exceed fifty lakh rupees (₹50,00,000).
State Consumer Disputes Redressal Commission (SCDRC): Handles cases where the total value claimed exceeds fifty lakh rupees (₹50,00,000), but does not exceed two crore rupees (₹2,00,00,000).
National Consumer Disputes Redressal Commission (NCDRC): Handles cases where the value of the goods or services along with the compensation claimed exceeds two crore rupees (₹2,00,00,000).
These jurisdictional thresholds ensure that consumer grievances are addressed efficiently at the appropriate level.” 
—
Question: “ Where can a consumer case be filed ?” 
Answer: “ A complaint must be filed in the District Consumer Disputes Redressal Commission (DCDRC) within whose local limits of jurisdiction one of the following applies:
The place of business or residence of the opposite party
The place of business or residence of the complainant
Where the cause of action, wholly or partially, arises”
—
Question: “ ​​Who can make a complaint?”
“A complaint can be made by various parties under the Consumer Protection Act. These include:
A Consumer: Any individual who has purchased goods or services for personal use.
Registered Voluntary Consumer Associations: Any consumer association registered under existing laws.
Central or State Government: Both levels of government can file complaints on behalf of consumers.
The Central Authority: The Central Consumer Protection Authority can initiate complaints to protect consumer rights.
Multiple Consumers (Class Action): Groups of consumers with the same interest can file a complaint collectively, with the permission of the consumer commission.
Legal Heirs or Representatives: In case of a consumer's death, their legal heir or representative can file a complaint.
Parents or Legal Guardians: For a minor consumer, their parent or legal guardian can file a complaint.
These provisions ensure that consumer grievances can be addressed effectively by allowing various representatives to seek redressal.”
—
Question: “What is the procedure to file a complaint before the consumer commission?”
Answer: “To file a complaint before the Consumer Commission, follow these steps:
Writing the Complaint:
The complaint should be in writing and contain all relevant details.
Specify the aggrieved consumer's details, the service provider or manufacturer's details, and the facts of the matter.
Clearly state the remedy sought.
Filing the Complaint:
Complaints can be filed in person or through an agent.
They can also be sent by registered post along with the required court fee.
Complaints can be filed online at the eJagriti portal: https://ejagriti.gov.in/.
Submission of Copies:
Normally, three copies of the complaint are required.
One copy is retained for official purposes, one is forwarded to the opposite party, and one is for the complainant.
If there are multiple opposite parties, more copies may be required.
Contents of the Complaint:
The complaint must contain all necessary details of the aggrieved consumer, the service provider, or manufacturer.
Include contact details, place of business, relevant facts of the matter, and the remedy sought.
Attach an affidavit signed and verified by the complainant.
Attach all other relevant documents, such as bill details, mode of payment, and guarantee or warranty cards.”
—
Question: “What is the procedure to file a complaint in the eJagriti platform?”
Answer: “ Following these steps will ensure that your complaint is filed correctly on the eJagriti platform.

Create an Account: If you don't have an account, register on the eJagriti portal at https://e-jagriti.gov.in/register.
Provide your Phone Number, Name, and a password for your eJagriti account. Confirm your password by entering it in the corresponding box, and then click ‘Role. Select ‘Consumer/Authorised Representative’. If you do not possess an Indian address and phone number, click on the ‘NRI’ check box at the top of the form.
Verify Phone Number/Email: Verify your phone number/email address by copying the six digit code sent to your registered email or phone number.
Verify Address: If you are not NRI, the portal will ask you for an address and ask you to upload an address proof. Please write down your address, select and upload a scanned copy of any government authorised address proof. 
Login: Use your credentials to log in to the portal.
Start New Case: Click on 'File a New Case' from the Filing dropdown menu and select 'Consumer Complaint'. Then click on the ‘NEXT’ button.
Enter Case Details in the respective boxes:
Paid as Consideration: the amount you have paid for a product or service
Claim Consideration: the amount you wish to claim as relief
Date of Cause of Action: enter the date where the issue first occurred or when it was first noticed
State of Cause of Action: enter the state where you wish to file the complaint
District of Cause of Action: enter the District where you wish to file the complaint
Case category: select the most appropriate category for your complaint
Click on the ‘NEXT’ button at the bottom
Enter Complainant and Opposite Party details in the respective boxes
Complainant details: enter the complainant name, and if you are not the complainant enter your relation to them and your name as the ‘Relative Name’
Enter the complainant mobile number and email address. Select if you/they fall under Senior citizen, widow, differently abled, or serious ailment categories
Enter address of the complainant
Enter the details of any advocate or legal representative whose services you are using, in case you have an advocate for this complaint
Enter the details of the Opposite part, similar to details of the complainant, including Address and Name of the business
Select the accurate State, and District for the address of the Opposite Party, and you may add any additional addresses as well
Select ‘NEXT’ option at the bottom of the form
Add details of any additional complainant parties in the “ADD ADDITIONAL COMPLAINANT” option. Clicking this option will drop down a fresh set of boxes for you to fill information. You may add as many additional complainants as necessary. If your complaint involves a ‘pro-forma’ complainant you may add their details here, under the “ADD PROFORMA COMPLAINANT DETAILS” option. Click ‘NEXT’ button at the bottom of the page once completed.
Add details of any additional Opposite Parties parties in the “ADD ADDITIONAL OPPOSITE PARTY” option. Clicking this option will drop down a fresh set of boxes for you to fill information. You may add as many additional complainants as necessary. If your complaint involves a ‘pro-forma’ complainant you may add their details here, under the “ADD PROFORMA OPPOSITE PARTY DETAILS” option. Click ‘NEXT’ button at the bottom of the page once completed.
Upload the required documents for filing your complaint. This includes the following documents:
Index: this is an index of all submitted documents
Proforma for filing consumer complaint: this is a copy of your filled our consumer complaint
Synopsis with List of Dates and Events: a summary of the facts of your complaint with a list of key dates relevant to the events of the complaint
Memo of Parties: a formatted document listing the parties involved in the complaint.
Consumer Complaint with Notarised Affidavit: a copy of your complaint, along with an affidavit attached, provided the affidavit has been authorised by a notary
Upload any other documents under the ‘ADD ANNEXURES / DOCUMENTS’ option
IA Documents: if you have any documents relating to an interim application (IA) you can upload them here.
Vakalatnama: upload a copy of the Vakalatnama hin this box. The Vakalatnama is a document where you have authorised your lawyer or legal representative to appear on your behalf for the matter
Select ‘NEXT’ option at the bottom of the screen
‘Your Case will be filed in District Commission’: select which District Commission you would like your complaint to be filed.
Before you do this, you may go back to the previous screens by clicking ‘PREVIOUS’ button on the bottom left of the page, and you may re-check all your documents to check that they are as required to you and for the Consumer Commission. If you do re-check the documents, you may once again reach this page as described in previous steps.
Click the box which says ‘I hereby declare that the information provided is true and correct’. Select ‘PREVIEW’ option at the bottom right of the page.
The page will show you the details and information included in your complaint. Check whether the details are all accurate and included as required. Once you are satisfied, you may click the ‘FINAL SUBMIT’ button at the bottom of the page.
You may now access the status of whether your complaint has been accepted by the Consumer Dispute Redressal Commission. You may do this by clicking on the ‘Pending’ button on the eJagriti dashboard page (https://e-jagriti.gov.in/consumer/dashboard)

—
Question: “Can a consumer complaint be resolved through mediation?” 

Answer: “Yes, a consumer complaint can be resolved through mediation. The relevant consumer commission, either at the first hearing after admission or later stages, may refer the complaint to mediation with the written consent of the parties if they believe there is potential for settlement. Parties can also opt for mediation at any stage of the complaint. If the dispute remains unresolved through mediation, the consumer commission will proceed to hear the case. Successful mediation results in a written agreement of the terms, while partial settlements are recorded, and the remaining issues are heard by the Commission. If mediation is unsuccessful, the Commission will pass an appropriate order within seven days of receiving the settlement report. There is no fee required for mediation. No appeal is permissible after the settlement of a dispute through mediation.”

Question: “ What are the contents of a consumer complaint particulars should be furnished in the complaint?” 

Answer: “ Contents of a consumer complaint:
Cause-title
Name, description, and address of the complainant
Name, description, and address of the opposite party or parties
Facts relating to the complaint and where and when it arose
Copies of written complaints and notices sent to the service provider
Explanation of how the opposite parties are liable and why they are answerable or accountable
Documents supporting allegations, with a list of these documents signed by the complainant
Specification of how the case falls within the jurisdiction of the tribunal
 Relief or remedy claimed by the complainant

Following is a illustrative draft for a consumer complaint, in LaTex format:

“

BEFORE THE DISTRICT CONSUMER DISPUTES REDRESSAL 
COMMISSION AT _________

CONSUMER COMPLAINT NO.     /20	  

IN THE MATTER OF: 

<Title and name of first complainant>,
<Address of first complainant>

<Title and name of second complainant>
<Address of second complainant>

<Title and name of third complainant and so on>
<Address of third complainant>
… COMPLAINANT
AND

<Title and name of first opposite party>,
<Address of first opposite party>

<Title and name of second opposite party>,
<Address of second opposite party>

<Title and name of third opposite party and so on>,
<Address of third opposite party and so on>

… OPPOSITE PARTIES

COMPLAINT UNDER SECTION 35 of THE CONSUMER PROTECTION ACT, 2019 RESPECTFULLY SHOWETH:

The address of the Complainant for the purpose of service of summons, notice etc is as shown in the cause title above and that of his counsel 

The address of the Opponents for the very purpose is the same as shown in the cause title above.

FACTS OF COMPLAINT: <Include facts about the complainant and the reason for the complaint and the other facts provided by the complainant. Include facts about the actions of the opposite party as presented by the complainant. Include the relief prayed for by the complainant. Keep each unique aspect of the complaint in one paragraph. The complaint may take anywhere from 4-12 paragraphs.>

TRANSACTION: <Should describe the transaction complained of i.e. particulars and details of goods/services availed, items of goods / kind and nature of service; date of purchase of goods/availing of service' amount paid as price/ consideration, full or in part towards the goods/service' Photocopies of the bill/cash memo/voucher or receipt should be attached and properly marked as Annexure- A, B, C, and so forth or 1, 2, 3 and so forth.>

CAUSE OF ACTION: The cause of action arose on <inlcude date of incident provided by the complainant/user> when <include incident giving rise to complaint> 
<Should explain the grievance, i.e. whether the loss of damage has been caused by some unfair trade practice or restrictive trade practice adopted by any trader or there is some defect in the goods or there has been deficiency in service or the trader has charged excessive price for the goods. Should elucidate the nature of deficiency or unfair trade practice adopted by the trader, i.e. relating to the quality of goods/services: sponsorship; warranty or guarantee for such period promised.>

RELIEF CLAIMED: <The nature and amount of relief claimed must be specified, and any specific performance from any party should be specified. Can include removal of defects in goods or deficiency in service; replacement with new goods; return of the price or charges or compensation on account of financial loss or injury>

JURISDICTION: <Mention the jurisdiction under which the complaint lies, according to: (i) the value of the claim (upto fifty lakh for DCDRC, upto two crore for SCDRC, rest to NCDRC),  and the geographic location of the cause of action or the opposite party.>

LIMITATION: That the present complaint is being filed within the period of limitation prescribed under section 69 of the Consumer Protection Act, 2019.

COURT FEE: As per Rule 7 of Consumer Protection (Consumer Dispute Redressal Commission) Rules 2020, the requisite court fee has been paid.

EVIDENCE:
<Should give details of documents and/or witnesses relied on to substantiate the case. The documents should be attached as Annexures. Incorporated in a list of evidence. List of witnesses (if any) may be filed similarly>
1. <mention a brief description of the first item of complainant evidence>
2. <mention a brief description of the second item of complainant party evidence>
3. <mention a brief description of the third item of complainant evidence, and so on>

PRAYER:
In the circumstances stated above, the Complainant prays that the Hon'ble Commission may be pleased to:
<mention a numbered list of all reliefs prayed for by the complainant.>


Complainant


PLACE:                                                                                                                         Signature
DATED:


VERIFICATION:
Herein I, <name and title of complainant>, the complainant above named, do hereby state on the solemn affirmation that the contents of the above paragraphs are read over and explained to me in my vernacular. The same are found to be true and correct to the best of my knowledge, belief, and information.

Name & signature of the complainant


”
—
Question: “Does a consumer need an advocate to represent their case before a commission?
Answer: “Consumers do not need an advocate to represent their case in the Consumer Commission. These commissions are designed to provide simple and speedy justice, free from the complexities of regular court procedures. The process is informal and does not require the involvement of an advocate or pleader. Consumers can file and represent their complaint themselves or through a representative.”
—
Question: “What is the time limit for filing the complaint?” 

Answer: “The time limit for filing a complaint is two years from the date the cause of action arises, as per Section 35 of the Act. This period starts from the day the deficiency in service or defect in goods is detected. It's also referred to as the limitation period. However, the law allows consumers to file complaints even after this period if the District Consumer Disputes Redressal Commission (DCDRC) deems the reasons for the delay genuine and valid.” 
_
Question: “How to make the payment of the Court fee?” 

Answer: “The court fee for every complaint filed before the District Consumer Disputes Redressal Commission (DCDRC), State Consumer Disputes Redressal Commission (SCDRC), or National Consumer Disputes Redressal Commission (NCDRC) must be accompanied by a fee specified. This fee should be in the form of a crossed Demand Draft drawn on a nationalized bank or through a crossed Indian Postal Order drawn in favor of the Registrar of the respective Commission. The payment should be payable at the place where the Commission is situated. The concerned District Consumer Disputes Redressal Commission (DCDRC) shall deposit the received amount.”
_
Question: “ What reliefs or remedies are provided by the consumer commission?”
Answer: “Consumer commissions can provide the following reliefs and remedies under the Consumer Protection Act, 2019:
Compensation: Grant compensation to consumers for harm caused by product liability issues.
Directive to Manufacturers, Service Providers, and Sellers: Issue directives to:
Remove defects from goods.
Replace goods.
Refund the price paid along with any interest.
Remove defects or deficiencies in services.
Award compensation for loss or injury.
Discontinue and not repeat unfair trade practices or restrictive trade practices.
Withdraw hazardous goods from being offered for sale.
Cease manufacture of hazardous goods and desist from offering hazardous services.
Pay a sum determined by the Commission for loss or injury suffered by a large number of consumers.
Issue corrective advertisement to neutralize the effect of misleading advertisement.
Provide adequate costs to parties.
These reliefs and remedies are aimed at protecting and compensating consumers for any harm caused by products or services.”
—
Question: “What are the details of fees to be paid while filing a complaint with the consumer commissions?”
Answer: “Yes, there is a fee for filing a complaint with the Consumer Commissions. The fee must be paid in the form of a crossed Demand Draft drawn on a nationalised bank or through a crossed Indian Postal Order drawn in favour of the Registrar of the relevant Commission and payable at the respective location where the Commission is situated. The payment may otherwise be through electronic mode as arranged by the specific Commission concerned.

The fee amount varies based on the value of goods or services paid as consideration, as specified in the table below:
District Consumer Disputes Redressal Commission (DCDRC):
Up to ₹5 Lakh: No Fee
Above ₹5 Lakh – Up to ₹10 Lakh: ₹200
Above ₹10 lakh – Up to ₹20 Lakh: ₹400
Above ₹20 Lakh – Up to ₹50 Lakh: ₹1000
State Consumer Disputes Redressal Commission (SCDRC): 
Above ₹50 Lakh – Up to ₹1 Crore: ₹2000
Above ₹1 Crore – Up to ₹2 Crore: ₹2500 
National Consumer Disputes Redressal Commission (NCDRC): 
Above ₹2 Crore – Up to ₹4 Crore: ₹3000 
Above ₹4 Crore – Up to ₹6 Crore: ₹4000 
Above ₹6 Crore – Up to ₹8 Crore: ₹5000 
Above ₹8 Crore – Up to ₹10 Crore: ₹6000
Above ₹10 Crore: ₹7500
The fee should be payable at the respective place where the District Consumer Disputes Redressal Commission (DCDRC), State Consumer Disputes Redressal Commission (SCDRC), or the National Consumer Disputes Redressal Commission (NCDRC) is situated, or through electronic mode as per the arrangement made by the concerned Commission.”
—
Question: “What if the consumer is not satisfied with the order of the Consumer commission?” 

Answer: “If a consumer is dissatisfied with the order of a Consumer Commission:
They can prefer an appeal within 30 days from the date of the order.
The appeal against the order of the District Consumer Disputes Redressal Commission (DCDRC) can be made before the State Consumer Disputes Redressal Commission (SCDRC).
The appeal against the order of the State Consumer Disputes Redressal Commission (SCDRC) can be made before the National Consumer Disputes Redressal Commission (NCDRC).
The appeal against the order of the National Consumer Disputes Redressal Commission (NCDRC) can be made before the Supreme Court.” 
—
Question: “ Is there any change in the amount to be deposited for filing an appeal?” 
Answer: For appeals before the State Consumer Disputes Redressal Commission (SCDRC) and National Consumer Disputes Redressal Commission (NCDRC), 50% of the total award amount passed by the lower commission needs to be deposited.” 
—
Question: “What documents are required while filing a complaint in a consumer commission?”
Answer: “Index: A list of all the documents attached, paginated and indexed.
List of Dates: A chronological list of events relevant to the complaint.
Memorandum of the Parties: Details of the complainant and the opposite party, including complete address and contact details.
Complaint with Notarized Affidavit: The complaint along with a notarized attested affidavit.
Requisite Fee: Payment proof of the fee required for filing the complaint.
Supporting Documents and Evidences: Receipts, vouchers, or any other evidence supporting the complaint. These documents should be attested as a True Copy on the last page with name and signature.
Application for Condonation of Delay: If there is a delay of more than 2 years from the cause of action, an application for the condonation of delay with a notarized and attested affidavit should be filed.
Ensure that all documents are uploaded in the specified format and meet the requirements of the eJagriti portal for smooth processing of the complaint.”
_ 
Question: “ How long does it take for a consumer case to be heard by a judge in a consumer commission?” 
Answer: “ The timeline for a consumer case to be heard by a judge in consumer commission varies. While the Consumer Protection Act, 2019 aims for resolution within three months, the actual duration can range from three months to over a year due to case complexities.” 
—
Question: “ What is the procedure for filing an appeal before the State Consumer Disputes Redressal Commission (SCDRC)? 
The Appellant or his authorized agent is required to file a Memorandum that is to be sent through registered post for acknowledgment to the State Consumer Disputes Redressal Commission (SCDRC). This memorandum must be typed in a legible manner with distinct heads, grounds of appeal without any argument or narrative. Such memorandums are to be accompanied by certified copy of the District Consumer Disputes Redressal Commission (DCDRC) order and the documents of a supportive nature to the appeal. Along with it, the appellant is required to submit four copies of the memorandum to the State Consumer Disputes Redressal Commission (SCDRC).

The appellant can only be heard or urged to be heard in support of the appeals that are made to the State Consumer Disputes Redressal Commission (SCDRC). However, the State Consumer Disputes Redressal Commission (SCDRC) is not required to confine itself within the appeal that is made. 
—
Questions: What is the time period within which the State Consumer Disputes Redressal Commission (SCDRC) needs to provide a decision for the appeal ? 
Answer: “The State Consumer Disputes Redressal Commission (SCDRC) must provide a decision for the appeal within 90 days from the date of admission. If the State Consumer Disputes Redressal Commission (SCDRC) decides to dispose of the case, adequate reasons for the decision must be provided in writing.” 
-
Question: “What are the different types of cases on e-Jagriti portal?”
Answer: The following are the case types on e-Jagriti portal:

CC : Consumer Complaint
FA : First Appeal
SA : Second Appeal
RP : Revision Petition
AE : Appeal Execution
CCPA Appeal: Central consumer Protection Authority Appeal
TA: Transfer Application
EA: Execution Application
RA: Review Application
CA: Caveat Application
MA: Miscellaneous Application 
IA:  Interlocutary Application
—
Questions: “What is the procedure to track the status of a case on e-Jagriti portal?”
Answer: “To track the status of a case on the e-Jagriti portal, follow these steps:
Visit the e-Jagriti portal at https://e-jagriti.gov.in/case-history-case-status
Enter the case details in the search field. 
Case Details can be searched in either of the two ways:
1) Enter Old Case Format 
Select Commission Level
State Commission and District Commission
Case Number (Format for old case number: )

2) Enter New Case Number Format / Filing Reference Number

Enter the Captcha code to get the information

Information on New Case Number Format
If old case number is CC/1/2024, then -
NCDRC - NC/CC/1/2024 (NC/Old Case Number)
SCDRC - SC/6/CC/1/2024 (SC/State LGD/Old Case Number)
DCDRC - DC/60/CC/1/2024 (DC/District LGD/Old Case Number)

—

Questions: "What is the procedure to track judgements on the e-Jagriti portal?"
To Track Judgement, visit 
https://e-jagriti.gov.in/judgement

Steps to Search for Judgements:

1. Search Field: Enter the relevant value in the search text field based on the selected parameter in the "Search By" dropdown.

2. Commission Level: Select the appropriate commission level — National, State, or District Level Commission — depending on where the case was filed.

3. State and District Commission: If the case pertains to a State or District Commission, select the respective State and District from the dropdown list.

4. Case Filing/Disposed: Choose between Filing or Disposed based on the case status.

5. Search By: Select one of the available parameters to search by:
   Case Number
   Complainant
   Respondent
   Complainant Advocate
   Respondent Advocate
   Industry Type

6. Order Type: Choose the relevant order type — Daily Order or Judgement.

7. Date Range: Specify the From Date and To Date corresponding to the selected Case Filing or Disposed parameters.
---
'''

general_corpus_prompt = "Here is the General Information on Consumer Laws in India. Use it to answer user queries.\n"+GENERAL_CORPUS

ending_prompt = '''Here is the Context:
{context}

The Central Consumer Protection Authority (CCPA) is not a forum for consumer complaint. It cannot be used to register complaints but just governs the bodies.

# Instructions 
- Ask only one question at a time and wait for the user's response before proceeding.
- Tailor your responses based on the information provided by the user.
- Request one detail at a time when asking for information (e.g., name, opponent). Avoid listing multiple questions or details all at once.
- Provide concise, relevant information at each step.
- Always be polite and professional in your interactions.
- Use only the following pieces of retrieved context to answer the question if giving out information.
- If user asks question that requires information like name, address, contact details, email address, phone number or any other personal information of organisations, companies or government bodies, give information only if it is present in the context
- If user asks information like address, contact details, email address, phone number or any other personal information of organisations, companies or government bodies that is not in context, tell that you do not have this information and suggest ways he can obtain this information.
- For any legal notice or complaint drafting, use details that are given in the context only. Use placeholders `[Address]` for any information not in context.
- Use only the facts/names provided in the context or by the user.
- Don't let the user know you answered the question using the context.
- All humans are eligible to file a complaint if they have purchased goods or services for personal use and have faced any issues with the product or service. Irrespective of the age, gender, caste, religion, profession, etc.
- Always give alternative to e-Jagriti for filing complaints, if any exists for the sector.
- Avoid requesting information that the user has already provided.


Answers questions solely based on the information provided. Do not rely on your pretraining knowledge or assumptions. If the information required to answer is not present in the given context, respond with: "I do not have enough information to answer this question."
'''
# EDIT BY KIRAN
# OLD code by Shrey

# - Give e-Jagriti as the primary platform for filing complaints unless the user asks for alternatives.

# def get_prompt(one_shot=False, general_corpus=False):
#     if one_shot and general_corpus:
#        return MAIN_SYSTEM_PROMPT.format(one_shot_example=one_shot_prompt, general_corpus=general_corpus_prompt,date=datetime.now().strftime("%Y-%m-%d")) + ending_prompt
#     elif one_shot:
#          return MAIN_SYSTEM_PROMPT.format(one_shot_example=one_shot_prompt, general_corpus="",date=datetime.now().strftime("%Y-%m-%d")) + ending_prompt
#     elif general_corpus:
#          return MAIN_SYSTEM_PROMPT.format(one_shot_example="", general_corpus=general_corpus_prompt,date=datetime.now().strftime("%Y-%m-%d")) + ending_prompt
#     else:
#         return MAIN_SYSTEM_PROMPT.format(one_shot_example="", general_corpus="",date=datetime.now().strftime("%Y-%m-%d")) + ending_prompt

# ================================ START =================================

sector_prompt = "Below is general information relevant to the sector {sector}.\n{sector_content}"

custom_one_shot_example_prompt = "Here is an example of how the chatbot would assist with consumer grievance. Adapt language, tone, and level of detail to match variations in user submissions while maintaining the core structure. Use it for only styling purposes for drafting Notice and Complaint. Although the examples show the chatbot assisting with drafting an Affidavit and MoP, you should refrain from doing so. This is a demo conversation.:\n\n {custom_one_shot_example}"

def get_prompt(one_shot=False, general_corpus=False, custom_one_shot_example=None, sector_content=None, sector=""):
    one_shot_content = custom_one_shot_example_prompt.format(custom_one_shot_example=custom_one_shot_example) if custom_one_shot_example else (one_shot_prompt if one_shot else "")
    general_content = general_corpus_prompt if general_corpus else ""
    sector_info = sector_prompt.format(sector=sector, sector_content=sector_content) if sector_content else ""
    return MAIN_SYSTEM_PROMPT.format(
        one_shot_example=one_shot_content,
        general_corpus=general_content,
        date=datetime.now().strftime("%Y-%m-%d"),
        sector_corpus=sector_info
    ) + ending_prompt

# ================================ END =================================