import qupath.lib.objects.PathObject
import qupath.lib.projects.ProjectImageEntry
import qupath.lib.gui.QuPathGUI

// Loop through all images in the project
def project = getProject()

for (entry in project.getImageList()) {
    // Read the image data without opening it in the viewer
    def imageData = entry.readImageData()
    def hierarchy = imageData.getHierarchy()
    
    // Get all annotation objects
    def annotations = hierarchy.getAnnotationObjects()
    
    // Unlock all annotations
    annotations.each { ann ->
        ann.setLocked(false)
    }
    
    // Remove all child objects for each annotation
    annotations.each { ann ->
        removeObjects(ann.getChildObjects(), false)
    }
    
    // Save the modified hierarchy back to the entry
    entry.saveImageData(imageData)
    
    print "Processed: " + entry.getImageName()
}

// Sync project changes
project.syncChanges()
print "All images processed and project saved."
