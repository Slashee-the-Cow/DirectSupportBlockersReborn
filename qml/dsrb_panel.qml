import QtQuick 6.0
import QtQuick.Controls 6.0
import QtQuick.Layouts 6.0

import UM 1.6 as UM
import Cura 1.1 as Cura

Item {

    id: directSupportsPanel

    UM.I18nCatalog {id: catalog; name: "directsupportsreborn"}
    
    function getCuraVersion(){
        if(CuraApplication.version){
            return CuraApplication.version()
        } else {
            return UM.Application.version
        }
    }

    function compareVersions(version1, version2) {
        const v1 = String(version1).split(".");
        const v2 = String(version2).split(".");

        for (let i = 0; i < Math.max(v1.length, v2.length); i++) {
            const num1 = parseInt(v1[i] || 0); // Handle missing components
            const num2 = parseInt(v2[i] || 0);

            if (num1 < num2) return -1;
            if (num1 > num2) return 1;
        }
        return 0; // Versions are equal
    }

    function isVersion57OrGreater(){
        //let version = CuraApplication ? CuraApplication.version() : (UM.Application ? UM.Application.version : null);
        let version = getCuraVersion()
        if(version){
            return compareVersions(version, "5.7.0") >= 0;
        } else {
            return False
        }
    }
    

    function getProperty(propertyName){
        if(isVersion57OrGreater()){
            return UM.Controller.properties.getValue(propertyName);
        } else {
            return UM.ActiveTool.properties.getValue(propertyName);
        }
    }

    function setProperty(propertyName, value){
        if(isVersion57OrGreater()){
            return UM.Controller.setProperty(propertyName, value);
        } else {
            return UM.ActiveTool.setProperty(propertyName, value);
        }
    }

    function triggerAction(action){
        if(isVersion57OrGreater()){
            return UM.Controller.triggerAction(action)
        } else {
            return UM.ActiveTool.triggerAction(action)
        }
    }

    function validateInt(test, min_value = -Infinity, max_value = Infinity){
        if (test === ""){return false;}
        let intTest = parseInt(test);
        if (isNaN(intTest)){return false;}
        if (intTest < min_value){return false;}
        if (intTest > max_value){return false;}
        return true;
    }

    function validateFloat(test, min_value = -Infinity, max_value = Infinity){
        if (test === ""){return false;}
        test = test.replace(",","."); // Use decimal separator computer expects
        let floatTest = parseFloat(test);
        if (isNaN(floatTest)){return false;}
        if (floatTest < min_value){return false;}
        if (floatTest > max_value){return false;}
        return true;
    }
    
    property var default_field_background: UM.Theme.getColor("detail_background")
    property var error_field_background: UM.Theme.getColor("setting_validation_error_background")

    function getBackgroundColour(valid){
        return valid ? default_field_background : error_field_background
    }

    function validateInputsBox(){
        let message = "";
        let box_width_valid = true;
        let box_depth_valid = true;
        let box_height_valid = true;

        if (!validateFloat(boxWidth, 0.1)){
            box_width_valid = false;
            message += catalog.i18nc("@error:box_width_invalid", "Box width must be at least 0.1mm.\n");
        }

        if (!validateFloat(boxDepth, 0.1)){
            box_depth_valid = false;
            message += catalog.i18nc("@error:box_depth_invalid", "Box depth must be at least 0.1mm.\n");
        }

        if (!blockerToPlate && !validateFloat(boxHeight, 0.1)){
            box_height_valid = false;
            message += catalog.i18nc("@error:box_height_invalid", "Box height must be at least 0.1mm if \"Blocker to plate\" is disabled.\n");
        }

        if (box_width_valid && box_depth_valid && (blockerToPlate || box_height_valid)){
            setProperty("BoxWidth", parseFloat(boxWidth))
            setProperty("BoxDepth", parseFloat(boxDepth))
            if (blockerToPlate){
                setProperty("BlockerToPlate", true)
            } else {
                setProperty("BoxHeight", parseFloat(boxHeight))
                setProperty("BlockerToPlate", false)
            }
            inputsValid = true
            setProperty("InputsValid", inputsValid)
        } else {
            inputsValid = false
            setProperty("InputsValid", inputsValid)
        }
        errorMessage = message
        boxWidthTextField.background.color = getBackgroundColour(box_width_valid)
        boxDepthTextField.background.color = getBackgroundColour(box_depth_valid)
        boxHeightTextField.background.color = getBackgroundColour(box_height_valid)
    }

    function validateInputsPyramid(){
        let message = "";
        let pyramid_top_width_valid = true;
        let pyramid_top_depth_valid = true;
        let pyramid_bottom_width_valid = true;
        let pyramid_bottom_depth_valid = true;
        let pyramid_height_valid = true;

        if (!validateFloat(pyramidTopWidth, 0.1)){
            pyramid_top_width_valid = false;
            message += catalog.i18nc("@error:pyramid_top_width_invalid", "Pyramid top width must be at least 0.1mm.\n");
        }

        if (!validateFloat(pyramidTopDepth, 0.1)){
            pyramid_top_depth_valid = false;
            message += catalog.i18nc("@error:pyramid_top_depth_invalid", "Pyramid top depth must be at least 0.1mm.\n");
        }

        if (!validateFloat(pyramidBottomWidth, 0.1)){
            pyramid_bottom_width_valid = false;
            message += catalog.i18nc("@error:pyramid_bottom_width_invalid", "Pyramid bottom width must be at least 0.1mm.\n");
        }

        if (!validateFloat(pyramidBottomDepth, 0.1)){
            pyramid_bottom_depth_valid = false;
            message += catalog.i18nc("@error:pyramid_bottom_depth_invalid", "Pyramid bottom depth must be at least 0.1mm.\n");
        }

        if (!blockerToPlate && !validateFloat(pyramidHeight, 0.1)){
            pyramid_height_valid = false;
            message += catalog.i18nc("@error:pyramid_height_invalid", "Pyramid height must be at least 0.1mm if \"Blocker to plate\" is disabled.\n");
        }

        if (pyramid_top_width_valid && pyramid_top_depth_valid && pyramid_bottom_width_valid && pyramid_bottom_depth_valid && (blockerToPlate || pyramid_height_valid)){
            setProperty("PyramidTopWidth", parseFloat(pyramidTopWidth))
            setProperty("PyramidTopDepth", parseFloat(pyramidTopDepth))
            setProperty("PyramidBottomWidth", parseFloat(pyramidBottomWidth))
            setProperty("PyramidBottomDepth", parseFloat(pyramidBottomDepth))
            if (blockerToPlate){
                setProperty("BlockerToPlate", true)
            } else {
                setProperty("PyramidHeight", parseFloat(pyramidHeight))
                setProperty("BlockerToPlate", false)
            }
            inputsValid = true
            setProperty("InputsValid", inputsValid)
        } else {
            inputsValid = false
            setProperty("InputsValid", inputsValid)
        }
        errorMessage = message
        pyramidTopWidthTextField.background.color = getBackgroundColour(pyramid_top_width_valid)
        pyramidTopDepthTextField.background.color = getBackgroundColour(pyramid_top_depth_valid)
        pyramidBottomWidthTextField.background.color = getBackgroundColour(pyramid_bottom_width_valid)
        pyramidBottomDepthTextField.background.color = getBackgroundColour(pyramid_bottom_depth_valid)
        pyramidHeightTextField.background.color = getBackgroundColour(pyramid_height_valid)
    }

        function validateInputsLine(){
        let message = "";
        let line_width_valid = true;
        let line_height_valid = true;

        if (!validateFloat(lineWidth, 0.1)){
            line_width_valid = false;
            message += catalog.i18nc("@error:line_width_invalid", "Line width must be at least 0.1mm.\n");
        }

        if (!blockerToPlate && !validateFloat(lineHeight, 0.1)){
            line_height_valid = false;
            message += catalog.i18nc("@error:line_height_invalid", "Line height must be at least 0.1mm if \"Blocker to plate\" is disabled.\n");
        }

        if (line_width_valid && (blockerToPlate || line_height_valid)){
            setProperty("LineWidth", parseFloat(lineWidth))
            if (blockerToPlate){
                setProperty("BlockerToPlate", true)
            } else {
                setProperty("LineHeight", parseFloat(lineHeight))
                setProperty("BlockerToPlate", false)
            }
            inputsValid = true
            setProperty("InputsValid", inputsValid)
        } else {
            inputsValid = false
            setProperty("InputsValid", inputsValid)
        }
        errorMessage = message
        lineWidthTextField.background.color = getBackgroundColour(line_width_valid)
        lineHeightTextField.background.color = getBackgroundColour(line_height_valid)
    }

    width: childrenRect.width
    height: childrenRect.height


    readonly property string boxBlockerType: "box_blocker_type"
    readonly property string pyramidBlockerType: "pyramid_blocker_type"
    readonly property string lineBlockerType: "line_blocker_type"
    readonly property string customBlockerType: "custom_blocker_type"

    property bool inputsValid: false
    property string errorMessage: ""

    property string currentBlockerType: boxBlockerType
    property bool blockerToPlate: false

    property string boxWidth: ""
    property string boxDepth: ""
    property string boxHeight: ""

    property string pyramidTopWidth: ""
    property string pyramidTopDepth: ""
    property string pyramidBottomWidth: ""
    property string pyramidBottomDepth: ""
    property string pyramidHeight: ""

    property string lineWidth: ""
    property string lineHeight: ""

    function updateProperties(){
        currentBlockerType = getProperty("BlockerType")
        blockerToPlate = getProperty("BlockerToPlate")

        boxWidth = getProperty("BoxWidth")
        boxDepth = getProperty("BoxDepth")
        boxHeight = getProperty("BoxHeight")

        pyramidTopWidth = getProperty("PyramidTopWidth")
        pyramidTopDepth = getProperty("PyramidTopDepth")
        pyramidBottomWidth = getProperty("PyramidBottomWidth")
        pyramidBottomDepth = getProperty("PyramidBottomDepth")
        pyramidHeight = getProperty("PyramidHeight")

        lineWidth = getProperty("LineWidth")
        lineHeight = getProperty("LineHeight")
    }

    function updateBlockerButtonState(type = null){
        if (type === null){
            type = currentBlockerType
        }
        boxButton.checked = type === boxBlockerType
        pyramidButton.checked = type === pyramidBlockerType
        lineButton.checked = type === lineBlockerType
        customButton.checked = type === customBlockerType
    }

    function setBlockerType(type){
        setProperty("BlockerType", type)
        currentBlockerType = type
        updateBlockerButtonState(type)
        switch (type){
            case boxBlockerType:
                settingsStackLayout.currentIndex = 0;
                Qt.callLater(validateInputsBox);
                break;
            case pyramidBlockerType:
                settingsStackLayout.currentIndex = 1;
                Qt.callLater(validateInputsPyramid);
                break;
            case lineBlockerType:
                settingsStackLayout.currentIndex = 2;
                Qt.callLater(validateInputsLine);
                break;
            case customBlockerType:
                settingsStackLayout.currentIndex = 3;
                break;
        }
    }

    function validateCurrentBlocker(){
        switch(currentBlockerType){
            case boxBlockerType:
                validateInputsBox();
                break;
            case pyramidBlockerType:
                validateInputsPyramid();
                break;
            case lineBlockerType:
                validateInputsLine();
                break;
        }
    }

    Component.onCompleted: {
        updateProperties()
        Qt.callLater(validateCurrentBlocker)
        setBlockerType(currentBlockerType)
    }

    property int textFieldMinWidth: 75
    property int labelMinWidth: 100

    ColumnLayout {
        id: mainColumn
        anchors.top: parent.top
        anchors.left: parent.left
        spacing: UM.Theme.getSize("default_margin").height
        /*UM.Label{
            text: "I'm an interface!"
        }*/

        Row { // Different blocker types
            id: blockerTypeButtons
            spacing: UM.Theme.getSize("default_margin").width/2

            UM.ToolbarButton {
                id: boxButton
                text: catalog.i18nc("@label", "Box support blocker")
                toolItem: UM.ColorImage {
                    source: Qt.resolvedUrl("box.svg")
                    color: UM.Theme.getColor("icon")
                }
                property bool needBorder: true
                checkable: true
                onClicked: setBlockerType(boxBlockerType)
                z: 4
            }

            UM.ToolbarButton {
                id: pyramidButton
                text: catalog.i18nc("@label", "Square / rectangular pyramid support blocker")
                toolItem: UM.ColorImage {
                    source: Qt.resolvedUrl("truncated_square_pyramid.svg")
                    color: UM.Theme.getColor("icon")
                }
                property bool needBorder: true
                checkable: true
                onClicked: setBlockerType(pyramidBlockerType)
                z: 3
            }

            UM.ToolbarButton {
                id: lineButton
                text: catalog.i18nc("@label", "Line support blocker")
                toolItem: UM.ColorImage {
                    source: Qt.resolvedUrl("line.svg")
                    color: UM.Theme.getColor("icon")
                }
                property bool needBorder: true
                checkable: true
                onClicked: setBlockerType(lineBlockerType)
                z: 2
            }

            UM.ToolbarButton {
                id: customButton
                text: catalog.i18nc("@label", "Custom support blocker")
                toolItem: UM.ColorImage {
                    source: Qt.resolvedUrl("custom.svg")
                    color: UM.Theme.getColor("icon")
                }
                property bool needBorder: true
                checkable: true
                onClicked: setBlockerType(customBlockerType)
                z: 1
            }
        }

        StackLayout{
            id: settingsStackLayout
            Layout.fillWidth: true
            currentIndex: 0

            GridLayout {
                id: boxBlockerControls
                columns: 2
                
                UM.Label{
                    text: catalog.i18nc("@controls:box", "Box Width")
                    Layout.minimumWidth: labelMinWidth
                }
                UM.TextFieldWithUnit{
                    id: boxWidthTextField
                    Layout.minimumWidth: textFieldMinWidth
                    height: UM.Theme.getSize("setting_control").height
                    unit: "mm"
                    text: boxWidth
                    validator: DoubleValidator{
                        decimals: 1
                        bottom: 0.1
                        notation: DoubleValidator.StandardNotation
                    }
                    onTextChanged: {
                        boxWidth = text
                        Qt.callLater(validateInputsBox)
                    }
                }
                UM.Label{
                    text: catalog.i18nc("@controls:box", "Box Depth")
                    Layout.minimumWidth: labelMinWidth
                }
                UM.TextFieldWithUnit{
                    id: boxDepthTextField
                    Layout.minimumWidth: textFieldMinWidth
                    height: UM.Theme.getSize("setting_control").height
                    unit: "mm"
                    text: boxDepth
                    validator: DoubleValidator{
                        decimals: 1
                        bottom: 0.1
                        notation: DoubleValidator.StandardNotation
                    }
                    onTextChanged: {
                        boxDepth = text
                        Qt.callLater(validateInputsBox)
                    }
                }
                UM.Label{
                    text: catalog.i18nc("@controls:box", "Box Height")
                    Layout.minimumWidth: labelMinWidth
                    visible: blockerToPlate != true
                }
                UM.TextFieldWithUnit{
                    id: boxHeightTextField
                    Layout.minimumWidth: textFieldMinWidth
                    height: UM.Theme.getSize("setting_control").height
                    unit: "mm"
                    text: boxHeight
                    validator: DoubleValidator{
                        decimals: 1
                        bottom: 0.1
                        notation: DoubleValidator.StandardNotation
                    }
                    onTextChanged: {
                        boxHeight = text
                        Qt.callLater(validateInputsBox)
                    }
                    visible: blockerToPlate != true
                }
            }
            GridLayout {
                id: pyramidBlockerControls
                columns: 2

                UM.Label{
                    Layout.columnSpan: 2
                    Layout.preferredWidth: 220
                    text: catalog.i18nc("@info:pyramid", "<b>Note:</b> With normal supports, supports will only be blocked on the part of the pyramid intersecting with the object.<br>Tree supports will avoid the whole pyramid.")
                }
                
                UM.Label{
                    text: catalog.i18nc("@controls:pyramid", "Pyramid Top Width")
                    Layout.minimumWidth: labelMinWidth
                }
                UM.TextFieldWithUnit{
                    id: pyramidTopWidthTextField
                    Layout.minimumWidth: textFieldMinWidth
                    height: UM.Theme.getSize("setting_control").height
                    unit: "mm"
                    text: pyramidTopWidth
                    validator: DoubleValidator{
                        decimals: 1
                        bottom: 0.1
                        notation: DoubleValidator.StandardNotation
                    }
                    onTextChanged: {
                        pyramidTopWidth = text
                        Qt.callLater(validateInputsPyramid)
                    }
                }
                UM.Label{
                    text: catalog.i18nc("@controls:pyramid", "Pyramid Top Depth")
                    Layout.minimumWidth: labelMinWidth
                }
                UM.TextFieldWithUnit{
                    id: pyramidTopDepthTextField
                    Layout.minimumWidth: textFieldMinWidth
                    height: UM.Theme.getSize("setting_control").height
                    unit: "mm"
                    text: pyramidTopDepth
                    validator: DoubleValidator{
                        decimals: 1
                        bottom: 0.1
                        notation: DoubleValidator.StandardNotation
                    }
                    onTextChanged: {
                        pyramidTopDepth = text
                        Qt.callLater(validateInputsPyramid)
                    }
                }
                UM.Label{
                    text: catalog.i18nc("@controls:pyramid", "Pyramid Bottom Width")
                    Layout.minimumWidth: labelMinWidth
                }
                UM.TextFieldWithUnit{
                    id: pyramidBottomWidthTextField
                    Layout.minimumWidth: textFieldMinWidth
                    height: UM.Theme.getSize("setting_control").height
                    unit: "mm"
                    text: pyramidBottomWidth
                    validator: DoubleValidator{
                        decimals: 1
                        bottom: 0.1
                        notation: DoubleValidator.StandardNotation
                    }
                    onTextChanged: {
                        pyramidBottomWidth = text
                        Qt.callLater(validateInputsPyramid)
                    }
                }
                UM.Label{
                    text: catalog.i18nc("@controls:pyramid", "Pyramid Bottom Depth")
                    Layout.minimumWidth: labelMinWidth
                }
                UM.TextFieldWithUnit{
                    id: pyramidBottomDepthTextField
                    Layout.minimumWidth: textFieldMinWidth
                    height: UM.Theme.getSize("setting_control").height
                    unit: "mm"
                    text: pyramidBottomDepth
                    validator: DoubleValidator{
                        decimals: 1
                        bottom: 0.1
                        notation: DoubleValidator.StandardNotation
                    }
                    onTextChanged: {
                        pyramidBottomDepth = text
                        Qt.callLater(validateInputsPyramid)
                    }
                }
                UM.Label{
                    text: catalog.i18nc("@controls:pyramid", "Pyramid Height")
                    Layout.minimumWidth: labelMinWidth
                    visible: blockerToPlate != true
                }
                UM.TextFieldWithUnit{
                    id: pyramidHeightTextField
                    Layout.minimumWidth: textFieldMinWidth
                    height: UM.Theme.getSize("setting_control").height
                    unit: "mm"
                    text: pyramidHeight
                    validator: DoubleValidator{
                        decimals: 1
                        bottom: 0.1
                        notation: DoubleValidator.StandardNotation
                    }
                    onTextChanged: {
                        pyramidHeight = text
                        Qt.callLater(validateInputsPyramid)
                    }
                    visible: blockerToPlate != true
                }
            }
            GridLayout {
                id: lineBlockerControls
                columns: 2
                
                UM.Label{
                    text: catalog.i18nc("@controls:line", "Line Width")
                    Layout.minimumWidth: labelMinWidth
                }
                UM.TextFieldWithUnit{
                    id: lineWidthTextField
                    Layout.minimumWidth: textFieldMinWidth
                    height: UM.Theme.getSize("setting_control").height
                    unit: "mm"
                    text: lineWidth
                    validator: DoubleValidator{
                        decimals: 1
                        bottom: 0.1
                        notation: DoubleValidator.StandardNotation
                    }
                    onTextChanged: {
                        lineWidth = text
                        Qt.callLater(validateInputsLine)
                    }
                }
                UM.Label{
                    text: catalog.i18nc("@controls:line", "Line Height")
                    Layout.minimumWidth: labelMinWidth
                    visible: blockerToPlate != true
                }
                UM.TextFieldWithUnit{
                    id: lineHeightTextField
                    Layout.minimumWidth: textFieldMinWidth
                    height: UM.Theme.getSize("setting_control").height
                    unit: "mm"
                    text: lineHeight
                    validator: DoubleValidator{
                        decimals: 1
                        bottom: 0.1
                        notation: DoubleValidator.StandardNotation
                    }
                    onTextChanged: {
                        lineHeight = text
                        Qt.callLater(validateInputsLine)
                    }
                    visible: blockerToPlate != true
                }
            }
            ColumnLayout{
                id: customBlockerControls
                spacing: UM.Theme.getSize("default_margin").height / 2
                UM.Label{
                    text: catalog.i18nc("@controls:custom", "Click the button below to turn the currently selected object into a support blocker.")
                    Layout.fillWidth: true
                    Layout.minimumWidth: labelMinWidth
                    Layout.maximumWidth: labelMinWidth + textFieldMinWidth
                    wrapMode: Text.Wrap
                }

                UM.Label{
                    text: catalog.i18nc("@controls:custom", "<b>Once you've made the blocker, you have to move the blocker so that it intersects the model which needs the blocker.</b>")
                    Layout.fillWidth: true
                    Layout.minimumWidth: labelMinWidth
                    Layout.maximumWidth: labelMinWidth + textFieldMinWidth
                    wrapMode: Text.Wrap
                }

                Cura.SecondaryButton{
                    id: makeCustomBlockerButton
                    Layout.preferredHeight: UM.Theme.getSize("setting_control").height
                    Layout.alignment: Qt.AlignHCenter
                    text: catalog.i18nc("@controls:custom", "Convert to blocker")
                    onClicked: triggerAction("convert_sceneNode_to_blocker")
                }

                UM.Label {
                    text: catalog.i18nc("@controls:custom", "<b>Note:</b> This will mess with the model geometry so that it might not slice as well if you turn it back into a regular model.<br>It is recommended you create a copy of your object as a backup if that is the case.")
                    Layout.fillWidth: true
                    Layout.minimumWidth: labelMinWidth
                    Layout.maximumWidth: labelMinWidth + textFieldMinWidth
                    wrapMode: Text.Wrap
                }
            }

            //Layout.preferredWidth: children.length > 0 ? children[currentIndex].implicitWidth : 0
            Layout.preferredHeight: children.length > 0 ? children[currentIndex].implicitHeight : 0
        }
        UM.CheckBox {
            id: blockerToPlateCheckBox
            text: catalog.i18nc("@controls:blocker_to_plate","Blocker to plate")
            checked: blockerToPlate
            onClicked: {
                blockerToPlate = checked
                setProperty("BlockerToPlate", checked)
            }
            visible: currentBlockerType != customBlockerType
        }
        UM.Label{
            id: errorDisplay
            Layout.fillWidth: true
            Layout.maximumWidth: 220
            visible: errorMessage != ""
            text: errorMessage
            color: UM.Theme.getColor("error")
            wrapMode: Text.Wrap
        }
    }
}