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
    
    def orderedClasses = [
        getPathClass('Head'),
        getPathClass('Torso'),
        getPathClass('Tail'),
        getPathClass('Wings')
    ]
    
    // Sort the list
    annotations.sort {a1, a2 -> Integer.compare(orderedClasses.indexOf(a1.getPathClass()), orderedClasses.indexOf(a2.getPathClass())) }
        
    print "Processed: " + entry.getImageName()
}

// Sync project changes
project.syncChanges()
print "All images processed and project saved."
