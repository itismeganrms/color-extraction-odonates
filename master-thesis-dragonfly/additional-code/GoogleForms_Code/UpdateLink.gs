function replaceHelpTextWithLink() {
  const form = FormApp.openById("10NWKqLXzPttTzvX_quiDJi0YARix9-CTdX1eDUxV_Ns");
  const items = form.getItems();

  // Your new sheet link here
  const newSheetLink = "https://itismeganrms.github.io/dragonfly-formulier/";
  const helpMessage = `Link for reference to key: ${newSheetLink}`

  items.forEach(item => {
    if (item.getType() === FormApp.ItemType.PAGE_BREAK) {
      const sec = item.asPageBreakItem();

      // Replace entire help text with just the link
      sec.setHelpText(helpMessage);
    }
  });

  Logger.log("All section help texts replaced with the new sheet link.");
}

