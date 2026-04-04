// function numberSectionsWithTotal() {
//   const form = FormApp.openById("10NWKqLXzPttTzvX_quiDJi0YARix9-CTdX1eDUxV_Ns"); // <-- Put your form ID here
//   const items = form.getItems();

//   // First count total number of sections
//   const totalSections = items.filter(it => it.getType() === FormApp.ItemType.PAGE_BREAK).length;

//   let sectionIndex = 0;

//   items.forEach(item => {
//     if (item.getType() === FormApp.ItemType.PAGE_BREAK) {
//       sectionIndex++;
//       let sectionItem = item.asPageBreakItem();
//       let title = sectionItem.getTitle().trim();

//       // Remove previous numbering if it exists
//       title = title.replace(/^Section\s+\d+\s*\/\s*\d+\s*\n+/i, "");

//       // Add new numbering with newline
//       const newTitle = `Section ${sectionIndex} / ${totalSections}\n${title}`;

//       sectionItem.setTitle(newTitle);
//     }
//   });

//   Logger.log(`Updated ${sectionIndex} sections`);
// }


// function cleanSectionHeaders() {
//   const form = FormApp.openById("10NWKqLXzPttTzvX_quiDJi0YARix9-CTdX1eDUxV_Ns");
//   const items = form.getItems();

//   // Count total sections (Page Breaks)
//   const totalSections = items.filter(i => i.getType() === FormApp.ItemType.PAGE_BREAK).length;
//   let sectionIndex = 0;

//   items.forEach(item => {
//     if (item.getType() === FormApp.ItemType.PAGE_BREAK) {
//       sectionIndex++;

//       let sec = item.asPageBreakItem();

//       // Original title & help text
//       // let title = sec.getTitle().trim();
//       let help = sec.getHelpText().trim();

//       // Remove old numbering like: "Section 4 / 1504. "
//       // title = title.replace(/^Section\s+\d+\s*\/\s*\d+\.?\s*/i, "");

//       const sheetLink = 'https://docs.google.com/spreadsheets/d/17T3n56Ws1WAdUNdmzz00Na3pjVL76IsXB2uLIaEbYOU/edit?usp=sharing';
//       const message = `Link for referring to the response key : ${sheetLink}`;

//       // Create new header line with *italic*
//       const newHeader = `Section ${sectionIndex} / ${totalSections}`;

//       // Build new description line:
//       // "Reference Sheet to Response Key : (link)"
//       // let newHelp = title;
//       if (help) {
//         help = message;
//       }

//       // Apply updates
//       // sec.setTitle(newHeader);
//       sec.setHelpText(help);
//     }
//   });

//   Logger.log("Section formatting complete.");
// }
