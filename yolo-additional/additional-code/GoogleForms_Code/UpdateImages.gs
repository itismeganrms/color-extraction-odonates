// function addImagesToForm() {
//   // 🔹 STEP 1: IDs for your Form and Drive Folder
//   const formId = '10NWKqLXzPttTzvX_quiDJi0YARix9-CTdX1eDUxV_Ns';       // e.g. from https://docs.google.com/forms/d/FORM_ID/edit
//   const folderId = '1epzvYV9dwtd41qOCjt0uaG1_7uJKLmLS';   // e.g. from https://drive.google.com/drive/folders/FOLDER_ID

//   const sheetLink = 'https://docs.google.com/spreadsheets/d/17T3n56Ws1WAdUNdmzz00Na3pjVL76IsXB2uLIaEbYOU/edit?usp=sharing';
//   const message = `Key for Responses (${sheetLink})`;

//   // 🔹 STEP 2: Open the Form and Folder
//   const form = FormApp.openById(formId);
//   const folder = DriveApp.getFolderById(folderId);

//   const files = folder.getFiles();
//   while (files.hasNext()) {
//     const file = files.next();

//     // Only process image files
//     if (file.getMimeType().startsWith('image/')) {
//       Logger.log('Adding image: ' + file.getName());

//       // Add the image to the form
//     // --- Create a section ---
//     form.addPageBreakItem().setTitle('Reference Sheet to Response Key').setHelpText(message);

//     const blob = file.getBlob();
//     let added = false;
//     let attempts = 0;

//     while (!added && attempts < 5) {
//       try {
//         // Add image item
//         form.addImageItem()
//           .setImage(blob);

//         // Add question
//         const item = form.addMultipleChoiceItem();
//         item.setTitle('Do you agree with the classification?')
//             .setChoices([
//               item.createChoice('Yes, only all seven parts (the head, thorax, abdomen and four wings) are present and recognised by the model'),
//               item.createChoice('Yes, all seven parts (the head, thorax, abdomen and four wings) are present and recognised by the model. In addition to this, there are other objects misrecognized by the model.'),
//               item.createChoice('Yes, the three main parts (the head, thorax and abdomen) are recognised, and there can be some discrepancies with the wings, and/or some misrecognised objects'),
//               item.createChoice("No, one or more of the three main parts (the head, thorax and abdomen) is not detected in the image"),
//               item.createChoice('No, one of the three main parts (the head, thorax and abdomen) is misclassified')
//             ]);

//         Logger.log('✅ Added image: ' + file.getName());
//         added = true;
//       } catch (e) {
//         attempts++;
//         Logger.log(`⚠️ Attempt ${attempts}: ${e.message}`);
//         Utilities.sleep(15000);
//       }
//     }

//     // Small delay between images
//     Utilities.sleep(2000);
//   }

//   Logger.log('🎉 All images added to form.');
// }
// }