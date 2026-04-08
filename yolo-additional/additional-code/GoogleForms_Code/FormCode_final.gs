// function addImagesToFormSections(formId, folderId) {
//   const form = FormApp.openById(formId);
//   const folder = DriveApp.getFolderById(folderId);
//   const files = folder.getFiles();

//   const sheetLink = 'https://docs.google.com/spreadsheets/d/17T3n56Ws1WAdUNdmzz00Na3pjVL76IsXB2uLIaEbYOU/edit?usp=sharing';
//   const message = `Key for Responses (${sheetLink})`;

//   let sectionIndex = 0;

//   while (files.hasNext()) {
//     const file = files.next();
//     if (!file.getMimeType().startsWith('image/')) continue;

//     sectionIndex++;

//     Logger.log(`🖼️ Processing: ${file.getName()}`);

//     // --- Create a section ---
//     form.addPageBreakItem().setTitle('Reference Sheet to Response Key').setHelpText(message);

//     // Add a section break
//     // form.addParagraphTextItem()
//     //   .setTitle('Reference sheet:')
//     //   .setHelpText(message);

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
//               item.createChoice('Yes, all seven parts (the head, thorax, abdomen and four wings) are present'),
//               item.createChoice('Yes, all seven parts (the head, thorax, abdomen and four wings) are present. In addition to this, there are other objects found and highlighted in the image'),
//               item.createChoice('Yes, the three main parts (the head, thorax and abdomen) are present. There are some discrepancies with the wings'),
//               item.createChoice('No, one of the three main parts (the head, thorax and abdomen) is absent or not detected in the image'),
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

// addImagesToFormSections(
//   formId = '10NWKqLXzPttTzvX_quiDJi0YARix9-CTdX1eDUxV_Ns',
//   folderId = '1GXZKiU2SZSg2Szi_koZeu_KLb0xSo9OA')