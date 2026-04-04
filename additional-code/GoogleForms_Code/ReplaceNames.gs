// function updateOptionText() {
//   // Open your form
//   var form = FormApp.openById('10NWKqLXzPttTzvX_quiDJi0YARix9-CTdX1eDUxV_Ns');
  
//   // Get all items in the form
//   var items = form.getItems();

//   // Loop through all items
//   for (var i = 0; i < items.length; i++) {
//     var item = items[i];
    
//     // Only modify multiple-choice items
//     if (item.getType() === FormApp.ItemType.MULTIPLE_CHOICE) {
//       var mcItem = item.asMultipleChoiceItem();
//       var choices = mcItem.getChoices();

//       var newChoices = choices.map(function(choice) {
//         var text = choice.getValue(); // current text
//         var newText = text;           // default to same text

//         // Replace only if it matches certain text
//         if (text === "Yes, all seven parts (the head, thorax, abdomen and four wings) are present") {
//           newText = "Yes, only all seven parts (the head, thorax, abdomen and four wings) are present and recognised by the model";
//         } else if (text === "Yes, all seven parts (the head, thorax, abdomen and four wings) are present. In addition to this, there are other objects found and highlighted in the image") {
//           newText = "Yes, all seven parts (the head, thorax, abdomen and four wings) are present and recognised by the model. In addition to this, there are other objects misrecognized by the model.";
//         } else if (text === "Yes, the three main parts (the head, thorax and abdomen) are present. There are some discrepancies with the wings") {
//           newText = "Yes, the three main parts (the head, thorax and abdomen) are recognised, and there can be some discrepancies with the wings, and/or some misrecognised objects";
//         } else if (text === "No, one of the three main parts (the head, thorax and abdomen) is absent or not detected in the image") {
//           newText = "No, one or more of the three main parts (the head, thorax and abdomen) is not detected in the image";
//         }

//         // Return a new choice (preserve correct-answer flag if used)
//         return mcItem.createChoice(newText, choice.isCorrectAnswer());
//       });

//       // Apply the updated choices
//       mcItem.setChoices(newChoices);
//     }
//   }

//   Logger.log("Options updated successfully!");
// }
