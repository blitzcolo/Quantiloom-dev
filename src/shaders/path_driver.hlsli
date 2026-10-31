#ifndef QUANTILOOM_PATH_DRIVER_HLSLI
#define QUANTILOOM_PATH_DRIVER_HLSLI

// ============================================================================
// The path driver: raygen shades the vertices of a path itself
// ============================================================================
// Included by raygen only. It walks the tree the recursion walked, depth
// first and in the same order, with the vertex library that was the
// closest-hit stage's body (ShadeVertexA / Advance / CompleteChild /
// FoldVertex) and the sky evaluator that was the miss stage's (ShadeMiss), so
// every draw is made from the stream state it was always made from.
//
// A ray traced from here is record-only (QlTraceRecord): closest-hit writes
// where it landed, miss says that it missed, and the vertex is shaded here.
// Every child is descended into the same way -- both environment-bounce
// lobes, the refracted or reflected ray, the hero-split ray and the three RGB
// dispersion rays, which Advance asks for, and a fusion interface's one child,
// which phase A asks for: the parent's frame is pushed, the child is shaded,
// and the parent resumes once the child's answer is in. Nothing is traced
// anywhere else -- closest-hit and miss only report where a ray landed -- so
// the pipeline's recursion depth is 1.
//
// One frame per ancestor of the vertex being shaded. A vertex pushes only to
// take a child, and every child is gated on depth < MAX_PATH_DEPTH: the bounce
// in SampleBounce, transmission where ShadeVertexA enables it, the interface
// child in BeginFusionInterface. Each child is one deeper than its parent, so
// the vertex at index i has depth i and at most MAX_PATH_DEPTH frames are
// live. That is the whole guard: with nothing left to recurse through, there
// is no other place a child could be traced, and a new kind of child must take
// the same gate.
//
// What a frame keeps beside the VertexFrame:
//   v       the vertex's own state as phase A left it, which the fold writes
//           its outputs into;
//   medium  the fusion medium statics as phase A left them. A child's phase A
//           overwrites them. The recursion never noticed, because each
//           closest-hit invocation had its own and every trace site saved and
//           restored them; here there is one set per path, so they are
//           restored before the parent resumes;
//   kind    the kind of the child in flight, which CompleteChild needs.
// ============================================================================

#include "shade_vertex.hlsli"
#include "shade_miss.hlsli"

struct PathFrame {
    VertexFrame         f;
    PathVertex          v;
    FusionMediumContext medium;
    uint                kind;
};

// Run a vertex on from where its frame stands: true when it asks for its next
// child (the request is in req), false once it has folded (its answer is in v).
bool QlRunVertex(inout VertexFrame f, inout PathVertex v, inout ChildRequest req) {
    if (Advance(f, v.depth, req)) {
        return true;
    }
    FoldVertex(f, v);
    return false;
}

// The whole path of one primary ray: vin is what the recursion's primary
// payload carried in, and the result is what it carried out.
//
// A freshly shaded vertex and a resuming one run on through separate calls of
// QlRunVertex on purpose. One shared call (9970461) made the frame and the
// vertex values carried around the loop from two sources, and measured slower
// in every render with a larger raygen stack, so it was reverted.
PathVertex RunPath(RayDesc ray, uint rayFlags, PathVertex vin) {
    PathFrame stack[MAX_PATH_DEPTH];
    uint sp = 0u;
    bool pathDone = false;
    [loop] while (!pathDone) {
        ChildRequest req = (ChildRequest)0;
        bool descend = false;

        // The vertex at the end of `ray`: phase A, then on until it asks for a
        // child or is finished. The hit ray is the one the closest-hit stage
        // reconstructed from its intrinsics when it still shaded.
        HitRecord rec;
        const bool hit = QlTraceRecord(ray, rayFlags, vin, rec);
        PathVertex fusionChild = (PathVertex)0;
        if (hit) {
            RayDesc hitRay = ray;
            hitRay.TMax = rec.t;
            VertexFrame f = (VertexFrame)0;
            ShadeVertexA(rec, hitRay, vin, f, req, fusionChild);
            if ((f.ctl & QL_CTL_FINISHED) == 0u) {
                // An interface vertex has asked for its child already; any
                // other vertex runs on to its first child or its fold.
                bool asks = (f.ctl & QL_CTL_FUSION) != 0u;
                if (!asks) {
                    asks = QlRunVertex(f, vin, req);
                }
                if (asks) {
                    stack[sp].f = f;
                    stack[sp].v = vin;
                    stack[sp].medium = SaveFusionMedium();
                    stack[sp].kind = req.kind;
                    ++sp;
                    descend = true;
                }
            }
        } else {
            ShadeMiss(vin, ray);
        }

        // Unless it asked for a child, vin is now a finished vertex. Fold it
        // into its parent, and the parent into its own, until one of them asks
        // for another child or the primary vertex is done.
        [loop] while (!descend && !pathDone) {
            if (sp == 0u) {
                pathDone = true;
            } else {
                const uint top = sp - 1u;
                RestoreFusionMedium(stack[top].medium);
                bool asks = false;
                if (stack[top].kind == QL_CHILD_FUSION) {
                    // An interface vertex's one child is in: fold it, and the
                    // vertex is done.
                    CompleteFusionInterface(stack[top].v, stack[top].f.fusion, vin);
                } else {
                    CompleteChild(stack[top].f, stack[top].kind, vin);
                    asks = QlRunVertex(stack[top].f, stack[top].v, req);
                }
                if (asks) {
                    stack[top].kind = req.kind;
                    descend = true;
                } else {
                    vin = stack[top].v;
                    sp = top;
                }
            }
        }

        if (descend) {
            ray = req.ray;
            rayFlags = req.rayFlags;
            if (req.kind == QL_CHILD_FUSION) {
                // Built whole by BeginFusionInterface, its medium included.
                vin = fusionChild;
            } else {
                // Built from the request, with the medium the statics hold,
                // which is the parent's at this point.
                vin = QlChildVertex(req);
            }
        }
    }
    return vin;
}

#endif // QUANTILOOM_PATH_DRIVER_HLSLI
