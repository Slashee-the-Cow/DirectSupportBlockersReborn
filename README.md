# Direct Support Blockers Reborn
*Easily create custom support blockers in Cura without being limited by starting with a silly little box.*  
  
Intended as a updated replacement for [Custom Support Eraser Plus](https://github.com/5axes/CustomSupportEraserPlus/) by 5@xes but not directly based on it.

## Features

### ~~Three~~ **Infinite** kinds of shapes to make support blockers out of!

#### **Box**
Prety much what it sounds like. Set the width, depth and height and click your object to create one, then move it where it's needed.  
(Technically it's called a cuboid, but that doesn't really roll off the tongue.)

#### **Square/Rectangular Pyramid**
Define sizes for the top and bottom (it can go from small to big or from big to small) and height.
- Normal supports will only avoid the part of the pyramid which intersects your model.
- Tree supports will avoid the whole pyramid. This is great to prevent them from growing in awkward places.

#### **Line**
Click two points on your model and it will draw a line of support blocker between them. Great if you're going to need access to the inner parts of something, or have a slight lip clean of supports.  

#### 🎆 **Custom** 🎆
**Turn any model you have in Cura into a support blocker!** Works super if you need thing really specific. (Which is sort of the point.) One thing I've done is if I have a model which is partially hollow, make a solid version of that and prevent any support generating inside.
- **Remember you'll need to move your custom blocker to intersect with the surface of the model that needs blocking.**
- This changes the model geometry in a way that might result in it not slicing as well if you turn it back into a regular model. If you think you might do that, please make a copy of your original.
- Converting very complex models into support blockers may make Cura appear to freeze (from a few seconds to several minutes in my "worst case scenario" tests) - please let it finish and Cura will start responding again.
- Cura's support generation may appear to ignore thin lines, that's because it can expand into those areas.  


All built in support types support being a fixed height or going down to the build plate.

---
### Get in touch!
Having any problems? Want to request a feature? Hear a good joke? Come say hi at the [GitHub repo](https://github.com/Slashee-the-Cow/DirectSupportBlockersReborn)!

---
### Known Issues
- Support blockers may be placed in the wrong location if there are any open messages in the bottom middle of the screen. This is a [bug in Cura](https://github.com/Ultimaker/Cura/issues/20488) which also affects the built in support blocker tool.
---
### Version History
#### 1.0.0
- Initial release.