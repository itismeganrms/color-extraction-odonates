def abdomen = getPathClass("tail")   // Your new class here
def staart = getPathClass("abdomen")

getAnnotationObjects().each { annotation ->
    if (annotation.getPathClass().equals(staart))
        annotation.setPathClass(abdomen)
}
//fireHierarchyUpdate() // If you want to update the count in the Annotation pane

print " Abdomen done!"

//def head = getPathClass("head")   // Your new class here
//def hoofd = getPathClass("hoofd")
//
//getAnnotationObjects().each{ annotation ->
//    if (annotation.getPathClass().equals(hoofd))
//        annotation.setPathClass(head)
//}
////fireHierarchyUpdate() // If you want to update the count in the Annotation pane
//
//print "Head done!"
//
//def thorax = getPathClass("torso")   // Your new class here
//def torso = getPathClass("thorax")
//
//getAnnotationObjects().each { annotation ->
//    if (annotation.getPathClass().equals(torso))
//        annotation.setPathClass(thorax)
//}
////fireHierarchyUpdate() // If you want to update the count in the Annotation pane
//
//print "Torso done!"