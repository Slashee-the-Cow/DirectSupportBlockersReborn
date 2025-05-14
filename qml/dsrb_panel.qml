import QtQuick 6.0
import QtQuick.Controls 6.0
import QtQuick.Layouts 6.0

import UM 1.6 as UM
//import Cura 1.1 as Cura

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
            setProperty("BoxLength", parseFloat(boxDepth))
            if (blockerToPlate){
                setProperty("BlockerToPlate", true)
            } else {
                setProperty("BoxHeight", parseFloat(boxHeight))
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

    width: childrenRect.width
    height: childrenRect.height


    readonly property string boxBlockerType: "box_blocker_type"
    readonly property string pyramidBlockerType: "pyramid_blocker_type"
    readonly property string lineBlockerType: "line_blocker_type"
    readonly property string customBlockerType: "custom_blocker_type"

    property bool inputsValid: false
    property string errorMessage: ""

    property string currentBlockerType: boxBlockerType
    property bool blockerToPlate: False

    property string boxWidth: ""
    property string boxDepth: ""
    property string boxHeight: ""


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
                Qt.callLAter(validateInputsCustom)
                break;
        }
    }

    property int textFieldMinWidth: 75

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
                text: catalog.i18nc("@label", "Square/rectangular pyramid support blocker")
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
                    text: catalog.i18nc("@controls:box_width", "Box Width")
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
                    text: catalog.i18nc("@controls:box_depth", "Box Depth")
                }
                UM.TextFieldWithUnit{
                    id: boxDepthhTextField
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
                    text: catalog.i18nc("@controls:box_height", "Box Height")
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
            Layout.maximumWidth: 175
            visible: errorMessage != ""
            text: errorMessage
            color: UM.Theme.getColor("error")
            wrapMode: Text.Wrap
        }
    }
}