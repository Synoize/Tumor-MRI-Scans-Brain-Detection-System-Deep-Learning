import React from 'react'
import { tools } from '../../assets/assets'
import Tool from '../../components/Tool'

const ToolsLayout = () => {
  return (
    <div className='container mx-auto h-screen overflow-y-scroll no-scrollbar px-4 md:px-12 py-6 bg-white'>
      <h1 className="text-2xl text-start text-gray-500">
        More <span className="text-special-red">Tools</span>
      </h1>

      <div className="grid grid-cols-2 md:grid-cols-6 md:gap-8 gap-4 my-6 ">
        {
          tools.map((tool, index) => (
            <Tool key={index} tool={tool}/>
          ))
        }
      </div>
    </div>
  )
}

export default ToolsLayout