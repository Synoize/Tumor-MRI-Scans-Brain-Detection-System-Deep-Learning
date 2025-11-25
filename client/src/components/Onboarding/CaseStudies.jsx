import React from "react";
import { MoveRight } from "lucide-react";
import { assets } from "../../assets/assets";
import { useAppContext } from "../../context/AppContext";

const CaseStudies = () => {
  const { navigate } = useAppContext();

  return (
    <section className="container mx-auto space-x-12 mb-20 px-4 md:px-12 flex lg:flex-row items-start flex-col-reverse text-black">
      {/* Left: Content */}
      <div className="w-full lg:w-1/2 space-y-4 px-4 md:px-0 md:pt-12">
        <h3 className="text-sm font-bold uppercase tracking-wide text-special-red">
          CASE STUDIES
        </h3>
        <h2 className="text-2xl md:text-5xl font-light leading-snug">
          Several case studies were conducted using MRI scan
        </h2>
        <p className="font-light md:text-xl">
          datasets consisting of Glioma, Meningioma, Pituitary tumors, and Normal
          (No Tumor) images. These case studies illustrate how the system
          performs across different tumor types and scenarios.
        </p>

        {/* Case Study Card */}
        <div className="p-4 border-l-4 border-special-red bg-slate-50 rounded-sm mt-8">
          <h2 className="text-xl">Glioma Tumor Detection</h2>
          <div className="text-start mt-2 flex flex-col space-y-1">
            <h3 className="font-light">Patient Information</h3>
            <div className="flex gap-10">
              <p>
                Age: <span className="font-light">45 years</span>
              </p>
              <p>
                Gender: <span className="font-light">Male</span>
              </p>
            </div>
            <p>
              Symptoms:{" "}
              <span className="font-light">
                Frequent headaches, nausea, and vision issues
              </span>
            </p>
          </div>
        </div>

        {/* CTA Button */}
        <div onClick={() => { navigate('/case-studies'); scrollTo(0, 0) }}
          className="inline-flex items-center justify-center px-6 py-3 bg-special-red/10 text-black rounded-md hover:bg-special-red/15 shadow-sm transition mt-12 group cursor-pointer"
        >
          More Studies
          <span className="ml-3 inline-flex items-center justify-center w-6 h-6 rounded-full text-sm group-hover:translate-x-2 transition-all duration-500">
            <MoveRight size={20} className="-rotate-45 group-hover:rotate-0 transition-all duration-500"/>
          </span>
        </div>
      </div>

      {/* Right: Image */}
      <div className="w-full lg:w-1/2 mb-12 md:mb-0">
        <img
          loading="lazy"
          src={assets.case_studies}
          alt="Team Meeting"
          className="w-full h-auto rounded-2xl object-cover"
        />
      </div>
    </section>
  );
};

export default CaseStudies;
